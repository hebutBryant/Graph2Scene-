"""Qwen-Image-2.1 adapter for its single-stream, cached-prefix attention."""

from pathlib import Path

import torch

from attention.controller import AttentionContext
from attention.store import AttentionRecord


MODEL_PATH = Path("/home/hdd4/lpz/Qwen-Image-2.1")


def load_pipeline():
    from diffusers import QwenImage21Pipeline

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL_PATH), dtype=torch.bfloat16)
    # The checkpoint has a large VLM and DiT; sequential offload fits one A6000.
    pipe.enable_sequential_cpu_offload()
    return pipe


class Qwen21AttentionProcessor:
    """Delegate native attention; reconstruct selected QK rows only when needed.

    For a future modifier, recompute target-image output in small query chunks.
    The native processor still executes first, preserving prefix KV extraction,
    causal prefill and all untouched rows. Prefix queries are never changed.
    """

    def __init__(self, layer, base, controller, store, chunk_size=64):
        self.layer = layer
        self.base = base
        self.controller = controller
        self.store = store
        self.chunk_size = chunk_size

    @torch.no_grad()
    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        out = self.base(
            attn, hidden_states, attention_mask=attention_mask,
            rotary_emb=rotary_emb, layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode, cache_write_slice=cache_write_slice,
            segments=segments, key_valid=key_valid,
        )
        capture = self.controller.should_capture(self.layer)
        modify = self.controller.should_modify(self.layer)
        if not capture and not modify:
            return out

        from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        if rotary_emb is not None:
            q = apply_rotary_emb_qwen(q, rotary_emb, use_real=False)
            k = apply_rotary_emb_qwen(k, rotary_emb, use_real=False)
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1)) if modify else None
        if kv_cache_mode == "cached":
            if layer_cache is None:
                raise RuntimeError("Qwen cached decode has no layer cache")
            cached_k, cached_v = layer_cache.get()
            k = torch.cat((cached_k, k), dim=1)
            if modify:
                v = torch.cat((cached_v, v), dim=1)
            prefix_len = k.shape[1] - q.shape[1]
            target_start = 0
        else:
            if not segments:
                raise RuntimeError("Qwen prefill segments unavailable; cannot identify target queries")
            prefix_len = segments[-1][1]
            target_start = prefix_len

        target_q = q[:, target_start:]
        if target_q.shape[1] == 0:
            return out
        valid = key_valid if key_valid is not None else attention_mask
        if valid is not None:
            if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
                raise RuntimeError("Expected Qwen boolean key-validity mask")
            valid = valid.reshape(valid.shape[0], -1)
            if valid.shape[1] != k.shape[1]:
                raise RuntimeError("Key-validity mask length does not match K")

        qh = target_q.transpose(1, 2).float()
        kh = k.transpose(1, 2).float()
        context_base = dict(
            layer=self.layer, step=self.controller.step,
            timestep=self.controller.timestep(), attention_type="target_joint",
            key_length=k.shape[1], prefix_length=prefix_len,
        )
        if capture:
            row_id = target_q.shape[1] // 2
            scores = torch.matmul(qh[:, :, row_id:row_id + 1], kh.transpose(-1, -2)) * (qh.shape[-1] ** -0.5)
            if modify:
                scores = self._modified(scores, context_base, row_id, row_id + 1)
            probs = self._probabilities(scores, valid)
            row = probs[0, :, 0].mean(dim=0).to(torch.float16).cpu()
            self.store.add(AttentionRecord(
                layer=self.layer, step=self.controller.step, timestep=context_base["timestep"],
                query_shape=tuple(target_q.shape), key_shape=tuple(k.shape),
                cross_shape=(target_q.shape[0], attn.heads, target_q.shape[1], prefix_len),
                self_shape=(target_q.shape[0], attn.heads, target_q.shape[1], k.shape[1] - prefix_len),
                prefix_mass=float(row[:prefix_len].float().sum()),
                target_mass=float(row[prefix_len:].float().sum()), sampled_row=row,
            ))
        if not modify:
            return out

        vh = v.transpose(1, 2)
        chunks = []
        for start in range(0, qh.shape[2], self.chunk_size):
            end = min(start + self.chunk_size, qh.shape[2])
            scores = torch.matmul(qh[:, :, start:end], kh.transpose(-1, -2)) * (qh.shape[-1] ** -0.5)
            scores = self._modified(scores, context_base, start, end)
            probs = self._probabilities(scores, valid)
            chunks.append(torch.matmul(probs.to(vh.dtype), vh).transpose(1, 2))
        target = torch.cat(chunks, dim=1).flatten(2, 3).to(q.dtype)
        target = attn.to_out[1](attn.to_out[0](target))
        result = out.clone()
        result[:, target_start:] = target
        return result

    def _modified(self, scores, context_base, start, end):
        context = AttentionContext(**context_base, query_start=start, query_end=end)
        updated = self.controller.modifier(scores, context)
        if updated.shape != scores.shape or updated.device != scores.device:
            raise ValueError("Attention modifier must return logits with the same shape and device")
        if not torch.isfinite(updated).all():
            raise ValueError("Attention modifier returned non-finite logits")
        return updated

    @staticmethod
    def _probabilities(scores, valid):
        if valid is not None:
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1)


def processor_factory(name, original, controller, store):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

    if name not in {controller.capture_layer, controller.modify_layer}:
        return None
    if not isinstance(original, QwenImage21AttnProcessor):
        raise RuntimeError(f"{name}: expected QwenImage21AttnProcessor, got {type(original).__name__}")
    return Qwen21AttentionProcessor(name, original, controller, store)
