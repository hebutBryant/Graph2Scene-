"""Reversible processor installation; model-specific math lives in the adapter."""

from contextlib import AbstractContextManager


class AttentionHook(AbstractContextManager):
    def __init__(self, transformer, controller, store, processor_factory):
        self.transformer = transformer
        self.controller = controller
        self.store = store
        self.processor_factory = processor_factory
        self.originals = {}

    def __enter__(self):
        try:
            for name, module in self.transformer.named_modules():
                if not name.endswith(".attn") or not hasattr(module, "set_processor"):
                    continue
                original = module.processor
                replacement = self.processor_factory(name, original, self.controller, self.store)
                if replacement is not None:
                    module.set_processor(replacement)
                    self.originals[name] = (module, original)
        except Exception:
            self.__exit__(None, None, None)
            raise
        if not self.originals:
            raise RuntimeError("No compatible attention processors found")
        print(f"[attention] installed on {len(self.originals)} layers", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        for module, original in self.originals.values():
            module.set_processor(original)
        self.originals.clear()
        return False
