# LLAISYS 扩展实现与实验报告

## 1. 说明

根据最开始给出的项目扩展要求，本次工作没有尝试一次性完成全部方向，而是选择了其中关联性最强、当前工程基础最好、且能够实际验证效果的几个方向继续完善。

本次实际完成的内容主要覆盖：

- 项目 #1：继续优化 LLAISYS 的 CPU 推理性能
- 项目 #3：实现支持采样、流式输出和交互式使用的聊天机器人
- 项目 #6：增加另一类模型的支持入口

## 2. 已完成任务

### 2.1 项目 #1：CPU 优化

已完成的 CPU 侧工作包括：

- 为 `linear` 增加 OpenMP 并行、分块、AVX2/FMA 快路径
- 为大规模 `f32` GEMM 接入 OpenBLAS
- 为 `linear` 增加启发式分发，而不是无条件走 BLAS
- 优化以下 CPU 算子：
  - `add`
  - `argmax`
  - `embedding`
  - `linear`
  - `rearrange`
  - `rms_norm`
  - `rope`
  - `self_attention`
  - `swiglu`
- 优化 Qwen2 模型推理路径中的 buffer 复用
  - decode 路径 buffer 复用
  - prefill 路径 scratch buffer 复用

其中 `linear` 是本次 CPU 优化的重点，也是最主要的性能收益来源。

### 2.2 项目 #3：AI 聊天机器人

已完成的聊天相关工作包括：

- 新增随机采样算子 `sample`
  - 支持 `temperature`
  - 支持 `top-k`
  - 支持 `top-p`
  - 当 `temperature <= 0` 或 `top_k == 1` 时回退到 argmax
- 将采样接入模型生成流程
- 新增 FastAPI 聊天服务
- 按 OpenAI chat-completions 风格实现 `POST /v1/chat/completions`
- 支持流式输出（SSE）
- 新增命令行交互式聊天 CLI

### 2.3 聊天会话管理

在项目 #3 的基础上，又继续补充了会话管理能力，已完成：

- 服务端会话存储
- 会话增删改查接口
  - `GET /v1/sessions`
  - `POST /v1/sessions`
  - `GET /v1/sessions/{session_id}`
  - `PUT /v1/sessions/{session_id}`
  - `DELETE /v1/sessions/{session_id}`
- 基于当前活动会话的前缀复用
- 回复重生成
- 编辑历史用户消息后重新生成
- 通过模型 `truncate` 接口回退 KV 状态

CLI 侧新增了这些会话命令：

- `/new`
- `/list`
- `/switch`
- `/history`
- `/regen`
- `/edit`
- `/delete`
- `/session`
- `/help`

### 2.4 项目 #6：支持新模型

增加了新的模型支持入口，当前已完成：

- Python 侧模型自动识别和自动加载
- 支持识别的模型类型：
  - `qwen2`
  - `llama`
  - `mistral`
- 新增 `create_model(model_path, ...)` 统一模型创建入口

这里需要明确说明：

- 当前新增模型支持主要是在 Python 侧做了兼容层和统一入口
- 底层后端仍然复用现有 decoder 风格实现
- 已完成 synthetic `llama` 用例验证
- 原有 Qwen2 真实模型推理路径也已重新验证通过

## 3. 关键实现说明

### 3.1 `linear` 的 BLAS 接入策略

在 OpenBLAS 接入过程中，并没有采用“所有大矩阵都走 BLAS”的做法。

实际测试发现，某些形状下无脑切换到 BLAS 会明显退化。因此最终采用了启发式策略：

- 仅在大规模 `f32` GEMM 且满足 `K >= N` 时优先走 OpenBLAS
- 其余场景继续走现有 AVX2/OpenMP 内核

这样可以保留 BLAS 在部分模型形状上的收益，同时避免 `1536 -> 8960` 这类扩张矩阵乘法退化。

### 3.2 会话编辑与重生成

为了支持聊天过程中的历史编辑和重生成，给模型增加了 `truncate` 能力，使服务端可以：

- 将活动会话的 KV 状态回退到指定位置
- 只对变化后的后缀重新计算

这比简单 `reset()` 然后整段历史重跑更合理，也更接近真实服务中的会话处理方式。

### 3.3 新模型支持方式

新模型支持没有直接复制一套新的完整后端，而是优先抽象了 Python 侧模型工厂：

- 读取 `config.json`
- 根据 `model_type` 自动选择模型包装类
- 复用现有公共加载逻辑和生成接口

这样可以先把模型接入路径打通，再决定后续是否需要为新模型单独扩展更底层的后端实现。

## 4. 实验环境

本次实验使用环境如下：

- 操作系统：Linux x86_64
- 构建工具：`xmake`
- Python 环境：`conda` 环境 `llaisys`
- 真实模型路径：
  - `/home/zyk/InfiniTensor/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- BLAS：
  - `OpenBLAS`
  - 安装位置：`/home/zyk/miniconda3/envs/llaisys`

## 5. 实验结果

### 5.1 初始基线

在最早的 CPU `linear` profile 中，曾观测到：

- `linear` 形状：`(64, 512) x (512, 512)`
- PyTorch：约 `0.269 ms`
- LLAISYS：约 `25.427 ms`

这也是后续优先优化 `linear` 的直接原因。

### 5.2 小规模 `linear` 优化结果

在 AVX2/OpenMP 优化后，短形状 `linear` 达到过：

- `linear` 形状：`(64, 512) x (512, 512)`
- LLAISYS：约 `0.0568 ms`

### 5.3 OpenBLAS 接入后的模型形状 `linear`

最终 BLAS 分发策略稳定后，针对更接近真实模型的矩阵形状，测得：

- `out=(198, 8960), x=(198, 1536), w=(8960, 1536)`
  - Torch：`4.88102 ms`
  - LLAISYS：`10.06528 ms`

- `out=(198, 1536), x=(198, 8960), w=(1536, 8960)`
  - Torch：`1.96279 ms`
  - LLAISYS：`4.23397 ms`

### 5.4 真实模型短生成

在 `DeepSeek-R1-Distill-Qwen-1.5B` 上，测得：

- 短 prompt，生成 8 个 token
  - `short_generate_s=0.600684`
  - `short_tok_per_s=13.318142`

### 5.5 长 prompt prefill + decode

在 prompt 长度约 198 tokens、只生成 1 个 token 的情况下，测得：

- `prefill_plus_decode_s=1.055171`

### 5.6 官方推理回归测试

运行：

```bash
python test/test_infer.py --device cpu --model /home/zyk/InfiniTensor/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 8
```

最近一次验证结果：

- Hugging Face：约 `0.71 s`
- LLAISYS：约 `0.72 s`
- token 序列：完全一致

## 6. 复现方式

### 6.1 构建

若当前环境中已经安装 OpenBLAS：

```bash
xmake f --cpu-blas=y --openblas-prefix="$CONDA_PREFIX" -cv
xmake
xmake install
```

### 6.2 核心回归测试

```bash
conda run -n llaisys env PYTHONPATH=python:test python test/test_runtime.py --device cpu
conda run -n llaisys env PYTHONPATH=python:test python test/ops/linear.py --device cpu
conda run -n llaisys env PYTHONPATH=python:test python test/ops/sample.py --device cpu
conda run -n llaisys env PYTHONPATH=python:test python test/test_generate_sampling.py --device cpu --model /home/zyk/InfiniTensor/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
conda run -n llaisys env PYTHONPATH=python:test python test/test_infer.py --device cpu --model /home/zyk/InfiniTensor/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --test --max_steps 8
```

### 6.3 聊天与会话管理测试

```bash
conda run -n llaisys env PYTHONPATH=python:test python test/test_chat_api.py
conda run -n llaisys env PYTHONPATH=python:test python test/test_chat_cli.py
```

### 6.4 新模型支持测试

```bash
conda run -n llaisys env PYTHONPATH=python:test python test/test_model_support.py
```

### 6.5 启动聊天服务

```bash
PYTHONPATH=python python -m llaisys.chat_server \
  --model-path /home/zyk/InfiniTensor/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu \
  --host 127.0.0.1 \
  --port 8000
```

### 6.6 启动交互式 CLI

```bash
PYTHONPATH=python python -m llaisys.chat_cli \
  --url http://127.0.0.1:8000 \
  --model llaisys-qwen2
```

常用会话命令：

- `/list`
- `/new`
- `/switch`
- `/history`
- `/regen`
- `/edit`

## 7. 当前说明

本次已经完成并验证的重点，是 CPU 路径优化、采样聊天、会话管理，以及新的模型支持入口。

当前仍未纳入本报告完成范围的方向包括：

- CUDA 运行时与 CUDA 算子
- 多用户推理服务与连续 batching
- 分布式推理

另外，新模型支持目前已经具备统一入口和 synthetic 验证，但底层后端仍主要复用现有 decoder 风格实现。原有 Qwen2 真实模型路径已经重新验证通过。
