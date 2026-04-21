# CLAUDE.md — openpi 快速参考

Physical Intelligence 的开源 VLA policy 库。**三款模型**(π0 flow / π0-FAST autoregressive / π0.5 flow+AdaRMS),**共享 PaliGemma 2B + SigLIP 视觉塔**,policy server + WebSocket + lightweight client 架构。

---

## 一句话架构

**client-server + openpi-client 通用库**。Server 跑 JAX/PyTorch 策略(`scripts/serve_policy.py`),client 只装 `openpi-client` pip 包通过 WebSocket 发 obs 收 action。同一个 server 能被真机 / 仿真 / dummy 任意 client 复用。

## 三款模型

| | **π0** | **π0-FAST** | **π0.5** |
|---|---|---|---|
| 总参数 | ~3.0B | ~2.7B | ~3.0B |
| 骨架 | PaliGemma 2B + Gemma 300M action expert | PaliGemma 2B | PaliGemma 2B + 300M expert(AdaRMS)|
| 动作表示 | flow matching | **autoregressive + FAST tokenizer** | flow matching |
| action horizon | 50 | 32 | 10 |
| bf16 权重 | ~6 GB | ~5.4 GB | ~6 GB |

**π0-FAST 是唯一 token-based 的**(action 离散化),π0/π0.5 直接 flow matching 连续动作。PyTorch 版本**不支持 π0-FAST**(只 JAX 可跑)。

## 发布的 checkpoint

Base(预训练,供 FT):`pi0_base` / `pi0_fast_base` / `pi05_base`

Fine-tuned(直接跑):

| ckpt | 机器人 | 备注 |
|---|---|---|
| **`pi05_droid_jointpos_polaris`** | DROID Franka | **π0.5 + 绝对关节控制,和 dreamzero Isaac Lab sim 组合已验证工作** |
| `pi0_fast_droid_jointpos_polaris` | DROID Franka | π0-FAST 变体,fallback 选项 |
| `pi0_droid_jointpos_polaris` | DROID Franka | π0 flow + jointpos |
| `pi0_fast_droid` / `pi0_droid` / `pi05_droid` | DROID Franka | 默认变体,action space 可能是 delta 而非 jointpos,和 sim 对接需测试 |
| `pi0_aloha_towel` / `pi0_aloha_tupperware` / `pi0_aloha_pen_uncap` | ALOHA 双臂 | 只适合真机 ALOHA,和 gym-aloha sim 的 cube 任务对不上 |
| **`pi05_libero`** | LIBERO(Franka)| **LIBERO SOTA 96.85%,最推荐的 sim 起步 ckpt** |

所有 `*_jointpos_polaris` 变体在 `src/openpi/training/misc/polaris_config.py`,ckpt 路径 `gs://openpi-assets/checkpoints/polaris/<config_name>/`。其他走 `gs://openpi-assets/checkpoints/<name>`,自动缓存到 `~/.cache/openpi`(可 `OPENPI_DATA_HOME` 改)。

> ⚠️ **老名字已废弃**:`pi0_fast_droid_jointpos`(无 `_polaris` 后缀)、`s3://openpi-assets-simeval/...` 路径在上游已移除。dreamzero 的 `run_sim_eval.py` docstring 里还写的是老名字,**以 `polaris_config.py` 为准**。

## 硬件需求(README:26-30)

| 模式 | VRAM | 3090 可行? |
|---|---|---|
| 推理 | >8 GB | ✅ 单卡就够(~8-10 GB 实际占用)|
| LoRA FT | >22.5 GB | ✅ 单卡边缘,**必设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`** |
| 全量 FT | >70 GB | ❌ 单卡装不下;双卡 FSDP 35GB/卡 也超 24GB,难 |

## 仿真环境(原生支持的只有 2 个)

| sim | 底层 | 有官方 ckpt | Isaac Sim? |
|---|---|---|---|
| **LIBERO** | MuJoCo + robosuite(EGL headless)| ✅ `pi05_libero` | 否 |
| ALOHA sim | MuJoCo + gym-aloha | ❌ FT ckpt 都是真机 | 否 |
| **DROID Isaac Lab** | **需配 dreamzero/eval_utils/run_sim_eval.py** | ✅ `pi0_fast_droid_jointpos` | **是(走 dreamzero)** |

⚠️ **openpi 本身没有任何 Isaac Sim/Isaac Lab 代码**。要跑 DROID sim 必须用 `dreamzero/eval_utils/run_sim_eval.py` 作 client。

## 两个"client"别混淆

1. **client 库** = `openpi-client` pip 包(`packages/openpi-client/`)。WebSocket + msgpack + image utils。两边 env 都装。
2. **client 应用** = 某个 main.py,驱动特定 sim/机器人。**不可互换**:
   - `examples/libero/main.py` — LIBERO sim
   - `examples/aloha_sim/main.py` — ALOHA sim
   - `examples/droid/main.py` — **真 DROID 硬件**(不是 sim!`from droid.robot_env import RobotEnv`)
   - `examples/aloha_real/main.py` — 真 ALOHA
   - `examples/simple_client/main.py` — dummy 测 server
   - `scripts/main.py`(`franka-scripts` 分支)— **实验室 Franka + RealSense,要 Python 3.8**
   - `dreamzero/eval_utils/run_sim_eval.py` — **Isaac Lab DROID sim,要 isaaclab3 env**

## 常用命令

### 起 server

```bash
# LIBERO 推荐起步
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
    --port 6000 policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

# Isaac Lab DROID sim 对接(本机已验证工作,端口 6000 必须)
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
    --port 6000 policy:checkpoint --policy.config=pi05_droid_jointpos_polaris \
    --policy.dir=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris

# 真机 Franka(franka-scripts 分支上 scripts/main.py)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
    policy:checkpoint --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

### 起 client(见"两种 client 应用"表)

- **LIBERO**(一键):`docker compose -f examples/libero/compose.yml up --build`
- **Isaac Lab DROID**:`conda activate env_isaaclab_legacy && CUDA_VISIBLE_DEVICES=0 python /home/zuxinrui/dreamzero/eval_utils/run_sim_eval.py --episodes 1 --scene 1 --host localhost --port 6000 --no-headless`
  - **GPU 必须 swap**:sim 用 GPU0(接显示器的那张),policy server 用 GPU1。双 3090 里只有接显示器的能跑 Vulkan swapchain
  - 用 `env_isaaclab_legacy`(Isaac Lab 2.0),**不要**用 isaaclab3(4.5)—— sim-evals 在 4.x 上 API 断裂
  - 详细部署和踩坑见 `docs/DROID_ISAAC_SIM_DEPLOYMENT.md`

### LoRA 微调(单 3090)

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi0_libero_low_mem_finetune --exp-name=my_exp --overwrite
```

LoRA 配置在 `src/openpi/training/config.py:679`(`paligemma_variant="gemma_2b_lora"`)。**π0.5 没官方 low_mem 配置**,要加自己写或走 FSDP 全量。

## 分支说明

- **`main`**:上游纯血(目前 HEAD = `650c5b0`)
- **`franka-scripts`**:`main` + 你从 `openpi_franka` port 过来的 7 个文件(`scripts/main.py`、`scripts/test*.py`、`scripts/readme.md`、`third_party/panda_python-*.whl`)。真机 Franka 实验用这个分支

## 关键文件速查

| 作用 | 路径 |
|---|---|
| 所有训练/推理 config | `src/openpi/training/config.py` |
| π0 / π0.5 模型 | `src/openpi/models/pi0_config.py`, `pi0.py` |
| π0-FAST 模型 | `src/openpi/models/pi0_fast.py` |
| Gemma/PaliGemma 骨架 | `src/openpi/models/gemma.py` |
| LIBERO 数据变换 | `src/openpi/policies/libero_policy.py` |
| DROID 数据变换(**关键**:确认 obs/action 格式)| `src/openpi/policies/droid_policy.py` |
| JAX 训练入口 | `scripts/train.py` |
| PyTorch 训练入口(**LoRA 不支持**)| `scripts/train_pytorch.py` |
| 推理 server 入口 | `scripts/serve_policy.py` |
| client 库源码 | `packages/openpi-client/src/openpi_client/` |
| WebSocket 协议文档 | `docs/remote_inference.md` |
| Franka 真机脚本(此分支)| `scripts/main.py`, `scripts/readme.md` |

## 环境与安装

- Python 3.11,`uv sync` 自动管依赖
- JAX 0.5.3 + CUDA 12,PyTorch 2.7.1(次等公民:无 LoRA / FSDP / mixed precision / π0-FAST)
- **PyTorch 用户必须 patch transformers**:`cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`(会污染 uv cache,撤销用 `uv cache clean transformers`)
- checkpoint 走 `gs://` 或 `s3://`,`fsspec[gcs]` 已包含

## 双 conda env 约定

| 任务 | env | 关键点 |
|---|---|---|
| policy server(本 repo 主体)| `openpi` uv `.venv`(Python 3.11)| `uv sync` 即可 |
| Isaac Lab sim client | **`env_isaaclab_legacy`**(Isaac Lab 2.0)| 用新版 isaaclab3 会因 mdp API 重组而失败 |
| LIBERO sim client | Python 3.8 独立 env | robosuite/MuJoCo 依赖对 3.11 不友好 |
| Franka 真机 client(franka-scripts 分支)| **Python 3.8 独立 env** | `panda_py` wheel 是 `cp38` only |

## 常见坑

- **`examples/droid/main.py` 不是仿真**,是真机 Franka(import `droid.robot_env`),没真机运行会直接挂
- **Isaac Sim DROID 仿真要用 dreamzero 的 `run_sim_eval.py`**,openpi 本身没有 Isaac 代码
- **Config 名上游漂移**:旧名 `pi0_fast_droid_jointpos` 已改成 `pi0_fast_droid_jointpos_polaris`,搬到 `src/openpi/training/misc/polaris_config.py`。ckpt 也从 `s3://openpi-assets-simeval/...` 换到 `gs://openpi-assets/checkpoints/polaris/...`
- `pi05_droid`(不带 `_jointpos_polaris` 后缀的默认版本)action space **可能是 delta EE** 而非绝对关节,直接丢 Isaac Sim 里机械臂会乱飞。用 `pi05_droid_jointpos_polaris` 才对
- **`--port 6000`** 必须显式写:`serve_policy.py` 默认 8000,`run_sim_eval.py` 默认 6000,两边要对齐
- **双 3090 开 `--no-headless` GUI 时必须用接显示器的 GPU**:sim 用 GPU0,policy 用 GPU1;Vulkan swapchain 只能在物理接 display 的 GPU 上建
- **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`** LoRA 训练必设,否则 JAX 默认 75% = 18GB,不够;sim 场景 0.5 足够,留空间给 Isaac Sim
- π0.5 在这个 repo 里**只支持 flow matching head**(不支持 FAST head)
- π0-FAST **只有 JAX 路径**,PyTorch 不支持
- `scripts/main.py`(Franka 真机)和 openpi 的 Python 3.11 env **不兼容**(`panda_py` wheel 是 cp38),client 必须独立 env
- gs:// 首次下载慢且无进度条,看 `~/.cache/openpi/` 的体积判断进度
- **Isaac Sim 启动 GUI 撞 `inotify` watcher 上限**:`fs.inotify.max_user_watches=524288` 写进 sysctl 永久修复
- **`pip install` 任何东西进 isaaclab3/env_isaaclab_legacy 都用 `--no-deps`**:Isaac Sim 硬 pin `numpy==2.3.1`,让 pip 自由解析 openpi-client deps 会把 numpy 降到 1.26.4 破坏 ABI

## 关于 dreamzero 这个邻居

`/home/zuxinrui/dreamzero` 里**唯一对 openpi 有用的东西** = `eval_utils/run_sim_eval.py`(Isaac Lab DROID sim client)+ `eval_utils/policy_client.py`(WebSocket 客户端,直接 import openpi-client 库)。**其他 dreamzero 代码都不在这条工作流里**。

本地 `run_sim_eval.py` 有一个 **未提交的 patch**(注释掉 `cv2.imshow`,因为 opencv-python-headless 无 GUI backend)。要撤回:`cd /home/zuxinrui/dreamzero && git checkout -- eval_utils/run_sim_eval.py`。

## 相关文档

- **`docs/DROID_ISAAC_SIM_DEPLOYMENT.md`** — DROID + Isaac Sim 完整部署 playbook + 所有踩过的坑(E1 inotify、E2 Vulkan、D2 USD 文件名漂移、D3 websockets 版本……)
- `docs/remote_inference.md` — WebSocket 协议
- `docs/docker.md` — Docker 化部署
