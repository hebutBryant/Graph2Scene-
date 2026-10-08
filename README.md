# Graph2Scene++：最小研究代码基座

当前阶段将 Graph2Scene 的 **attention hook → 少量 attention 捕获 → 可选 logits 修改** 机制接到 Qwen-Image-2.1。运行入口只接受文字 prompt，完成一次生成并保存图片。代码没有实现 scene graph 路由、多图参考或后续 Graph2Scene++ 算法。

## 环境与模型

- Python 3.10、支持 CUDA 的 PyTorch 和 NVIDIA GPU。
- 模型权重固定从 `/home/hdd4/lpz/Qwen-Image-2.1` 加载；该目录应包含 `model_index.json`、`transformer/`、`text_encoder/`、`processor/`、`scheduler/` 和 `vae/`。
- `requirement.txt` 记录了本机已验证的版本。尤其是 diffusers 固定到包含 `QwenImage21Pipeline` 的 Git 提交。
- 模型较大，加载时启用 sequential CPU offload；需要足够的主机内存与 GPU 显存。已在本机 `uno_env` 环境完成测试。

已有本机环境时：

```bash
conda activate uno_env
```

新建环境时：

```bash
conda create -n graph2scenepp python=3.10 -y
conda activate graph2scenepp
python -m pip install -r requirement.txt
```

安装 PyTorch 时如需指定 CUDA wheel，请先按机器驱动选择对应的 PyTorch 安装源，再安装其余依赖。这里不下载模型权重。

## 运行

在项目根目录执行：

```bash
python run.py --prompt "A girl gives a balloon to a boy."
```

默认参数为 512×512、20 步、seed 42；图片保存到 `outputs/sample.png`。快速检查可用：

```bash
python run.py \
  --prompt "A girl gives a balloon to a boy." \
  --steps 2 --width 256 --height 256 \
  --output outputs/smoke.png
```

`--capture-layer` 选择 DiT 层（0–31），`--capture-step` 选择从 0 开始计数的 denoising 步。例如读取第 1 步的缓存解码 attention：

```bash
python run.py \
  --prompt "A girl gives a balloon to a boy." \
  --steps 2 --width 256 --height 256 \
  --capture-layer 0 --capture-step 1 \
  --output outputs/step1.png
```

运行时会打印层名、step、scheduler timestep、Q/K 形状，以及 `target→prefix/cross` 和 `target→target/self` 的 attention 形状。只在指定层和步读取一个目标图像 query 的 attention 行，平均 head 后以 CPU float16 保存在运行进程的 `AttentionStore` 中；默认不会把全部 attention tensor 写入磁盘。图片是默认保存的产物。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `run.py` | 参数、Qwen 推理、逐步更新 controller、保存图片 |
| `models/qwen_image.py` | 加载本地模型；适配 Qwen 单流 attention、前缀 KV cache 和可选 logits 修改 |
| `attention/hook.py` | 安装 processor 并在退出时恢复原 processor |
| `attention/controller.py` | 指定捕获层／步和可选修改层／步窗 |
| `attention/store.py` | 保存少量采样记录并打印形状 |

Qwen-Image-2.1 的文本与图像 token 在同一条 attention 流中。纯文本生成时，目标图像 query 对前缀的 attention 是文字条件读取，对目标图像 token 的 attention 是图像自注意力。首步计算完整序列；后续步使用前缀 KV cache。因此这里没有沿用 FLUX 的双流／单流 processor 或固定的 `[text | image]` 偏移假设。

## 修改 attention 的入口

普通 `run.py` 只读取 attention，生成计算仍交给 Qwen 原 processor。研究代码可向 `AttentionController` 传入 `modifier(scores, context)`：

```python
def modifier(scores, context):
    # scores: [batch, heads, 当前 query 块, 全部 key]，为 softmax 前的 logits
    scores = scores.clone()
    scores[..., :context.prefix_length] += 0.2
    return scores

controller = AttentionController(
    capture_layer="transformer_blocks.0.attn",
    capture_step=0,
    modifier=modifier,
    modify_layer="transformer_blocks.0.attn",
    modify_start_step=0,
    modify_end_step=0,
)
```

将该 controller 传给 `AttentionHook`，并像 `run.py` 一样在 `callback_on_step_end` 中调用 `controller.set_step(step + 1)`。adapter 将目标图像 query 分块处理，在回调后应用原有的有效 key mask；前缀 attention 和 KV cache 保持原生行为。示例里的 `+0.2` 仅展示接口，不代表 Graph2Scene 路由算法。

## 已验证与范围

本机已完成 512×512、20 步普通生成，256×256、2 步的首步与缓存步 attention 捕获，以及真实 Qwen 模型上的 logits 修改推理。`run.py` 要求至少 2 步；本地 FlowMatch scheduler 在仅 1 步时会产生非有限值。

当前仅有 attention 干预基座。scene graph 的节点／边、bbox、token 对齐与时间调度尚未迁移到 Qwen 的联合序列；不要将示例中的 `target→prefix` 统计当作完整的 Graph2Scene 控制结果。
