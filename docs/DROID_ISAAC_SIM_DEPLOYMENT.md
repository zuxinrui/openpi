# DROID Isaac Sim + π0.5 Policy 部署踩坑全记录

在 Ubuntu 24.04 + dual RTX 3090 + NVIDIA driver 590.48.01 本机,把 **openpi 的 pi0.5-DROID 策略**通过 WebSocket 连上 **dreamzero 的 Isaac Lab sim client(基于 sim-evals)**,进行端到端仿真评估的**完整踩坑记录 + 最终可工作 playbook**。

本文档既是操作手册,也是"为什么当初这么搞"的备忘——六个月后自己回来能快速上手。

---

## 1. 目标与最终架构

```
GPU0 (has display)                        GPU1 (compute only)
env_isaaclab_legacy (conda, Py3.11)       openpi/.venv (uv, Py3.11)
─────────────────────────────────         ─────────────────────────────────
run_sim_eval.py (dreamzero)    ◄──────────  serve_policy.py (openpi)
  + Isaac Lab 2.0                           + JAX + PaliGemma 2B
  + sim_evals (DROID env)                   + pi05_droid_jointpos_polaris
  + Vulkan swapchain → 屏幕窗口              + gs:// checkpoint 下载
         │                                          ▲
         │ obs dict (DROID format)                  │
         │   observation/exterior_image_0_left      │
         │   observation/joint_position             │
         │   observation/gripper_position           │
         │   prompt                                 │
         ▼                                          │
         websocket ws://localhost:6000 ─────────────┘
         (openpi-client + msgpack_numpy)
         
         ◄───── action chunk (shape [15, 8]) ─────
                (7 arm joints + 1 gripper, abs joint pos)
```

## 2. 三个 repo 的角色

| repo | 位置 | 用途 | 分支 |
|---|---|---|---|
| **openpi** | `/home/zuxinrui/openpi` | policy server + 训练 + `openpi-client` 通用库 | `franka-scripts`(main + Franka port + CLAUDE.md) |
| **dreamzero** | `/home/zuxinrui/dreamzero` | `eval_utils/run_sim_eval.py`(sim client)+ `eval_utils/policy_client.py`(WebSocket wrapper) | `main`(本地有 cv2.imshow 补丁) |
| **sim-evals** | `/home/zuxinrui/sim-evals` | Isaac Lab DROID 环境定义 + 场景资产 | `main`(assets 下载后 + my_droid.usdz symlink) |

另有历史存档 `/home/zuxinrui/openpi_franka`(旧 fork,已把 Franka 脚本 port 到 openpi 的 `franka-scripts` 分支,原封不动保留作参考)。

## 3. 两个 conda env 的分工

| env | 位置 | Python | 主要用途 | 关键包 |
|---|---|---|---|---|
| **openpi 的 .venv** | `/home/zuxinrui/openpi/.venv`(uv 管理) | 3.11 | 跑 openpi policy server | JAX 0.5.3, PaliGemma, orbax, tyro |
| **env_isaaclab_legacy** | `~/miniconda3/envs/env_isaaclab_legacy` | 3.11 | 跑 Isaac Lab sim client | Isaac Lab 2.0(`/home/zuxinrui/IsaacLab-2.0`), isaacsim 5.1, sim-evals, openpi-client, websockets ≥13 |

**isaaclab3 env(新 Isaac Lab 4.5.22)绕开不用** —— sim-evals pin 2.2.0,4.x 的 `mdp` 模块重组后 API 断裂。

真机 Franka(不在本文档范围)用 Python 3.8 独立 env + `third_party/panda_python-*.whl`。

## 4. 最终可工作的启动命令(验证过,复制即可)

### Terminal 1 — openpi policy server(openpi .venv,**GPU1**)

```
cd /home/zuxinrui/openpi && CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py --port 6000 policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
```

出现 `Listening on 0.0.0.0:6000` 即 ready。首次会下 ~7GB ckpt。

### Terminal 3 — Isaac Lab sim client(env_isaaclab_legacy,**GPU0**,接显示器的那张)

```
conda activate env_isaaclab_legacy && cd /home/zuxinrui/openpi && CUDA_VISIBLE_DEVICES=0 python /home/zuxinrui/dreamzero/eval_utils/run_sim_eval.py --episodes 1 --scene 1 --host localhost --port 6000 --no-headless
```

`--no-headless` 打开 Isaac Sim GUI 实时看机械臂动作;想 batch 评估改 `--headless --episodes 10`。

视频输出在 `/home/zuxinrui/openpi/runs/<date>/<time>/episode_*.mp4`(因为 cwd 是 openpi,`run_sim_eval.py` 用相对 `Path("runs")`)。

### Scene 选择

| `--scene` | 任务 | 指令 |
|---|---|---|
| 1 | cube-in-bowl | put the cube in the bowl |
| 2 | can-in-mug | pick up the can and put it in the mug |
| 3 | banana-in-bin | put the banana in the bin |

## 5. 踩坑全纪录(按阶段)

下面每条 = 一个真实发生过的错误和它的修复。以后再撞同类问题能快速对号入座。

### 阶段 A:安装与环境

#### A1. `uv` 没装 → `Command 'uv' not found`

**别用 snap**(版本落后)。官方脚本:

```
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc && uv --version
```

#### A2. VS Code 自动激活 .venv + `conda activate` 叠加导致 Python 错乱

VS Code 打开 openpi 目录会自动 `source .venv/bin/activate`,再 `conda activate isaaclab3` 会把 conda env 叠加到 PATH 最前,但 `VIRTUAL_ENV` 还指着 .venv,状态混乱。

**干净做法**:开两个独立 terminal,或先 `deactivate` 再 `conda activate`。也可 `Ctrl+,` 关掉 `python.terminal.activateEnvironment`。

#### A3. `pip install openpi-client` 把 isaaclab3 的 numpy 降级(ABI 灾难)

openpi-client pyproject 里写 `numpy<2.0.0,>=1.22.4`,Isaac Sim 硬 pin `numpy==2.3.1`。pip 默认解析会把 numpy 降到 1.26.4 → Isaac Sim native extension ABI 不兼容。

**原则**:任何已有 Isaac Sim env 里装 pip 包都用 `--no-deps`。

```
# 修复
pip install "numpy==2.3.1" --force-reinstall
# 以后装东西
pip install -e /path/to/openpi/packages/openpi-client --no-deps
pip install -e /path/to/sim-evals --no-deps
```

#### A4. UV 的 `.venv` 会不会泄漏到其他项目?

`.venv/` 在 repo 本地,per-project 隔离。但 **uv 默认 hardlink mode** —— 改动 `.venv/lib/.../transformers/` 的文件会回溯污染 `~/.cache/uv/`,影响其他项目。

```
# 永久安全策略
export UV_LINK_MODE=copy  # 写到 ~/.bashrc
```

openpi README 让你 `cp ./src/openpi/models_pytorch/transformers_replace/* .venv/...` —— **我们没做这步**,因为 JAX 路径不需要 PyTorch。

### 阶段 B:上游配置漂移

#### B1. Checkpoint config 被重命名

- **老名**:`pi0_fast_droid_jointpos`(dreamzero docstring 里写的)
- **新名**:`pi0_fast_droid_jointpos_polaris`(加了 `_polaris` 后缀,搬到 `misc/polaris_config.py`)
- **意外福利**:新增了 `pi05_droid_jointpos_polaris`(π0.5 的 jointpos 变体),就是我们最终用的

三个可用变体,所有 ckpt 在 `gs://openpi-assets/checkpoints/polaris/<config_name>/`:

| Config | 模型 | action horizon |
|---|---|---|
| `pi05_droid_jointpos_polaris` | π0.5 flow matching | 15 |
| `pi0_fast_droid_jointpos_polaris` | π0-FAST autoregressive | 10 |
| `pi0_droid_jointpos_polaris` | π0 flow matching | (查 polaris_config.py) |

#### B2. 端口 6000 vs 8000

- `scripts/serve_policy.py` 的官方 README 示例用默认 8000
- `eval_utils/run_sim_eval.py` 的 `DreamZeroJointPosClient` 默认 **6000**

两边必须 `--port` 明确指定。本 playbook 全用 6000。

#### B3. simple_client 默认是 ALOHA 格式,对 DROID server 会被拒

```python
# simple_client/main.py:41
env: EnvMode = EnvMode.ALOHA_SIM  # 默认
```

DROID 策略 `DroidInputs` 要 `observation/gripper_position` 等 key,ALOHA 格式没有 → `KeyError` 。

**修正**:`uv run examples/simple_client/main.py --host localhost --port 6000 --env DROID`

但更实用的是跳过 simple_client,直接上 sim client —— sim 那边发的就是 DROID 格式。

### 阶段 C:Isaac Lab 版本

#### C1. Isaac Lab 4.5 和 sim-evals 的 2.2.0 pin 之间 API 断裂

症状:
```
AttributeError: No isaaclab.envs.mdp attribute observations
```

sim-evals 用 `mdp.observations.image`,在 Isaac Lab 2.x 存在,**4.x 里被扁平化重组**。

**修复**:用 env_isaaclab_legacy(Isaac Lab 2.0,对 2.2.0 的 API 差异很小,完全兼容 sim-evals)。**不要**碰 isaaclab3(Isaac Lab 4.5)。

检查某个 env 的 Isaac Lab 版本:
```
python -c "import isaaclab; print(isaaclab.__version__)"
```

#### C2. `pxr` 导入时机:必须在 AppLauncher 之后

```python
# run_sim_eval.py:149-162 的正确顺序
from isaaclab.app import AppLauncher          # ← 这步不触发 pxr
...
app_launcher = AppLauncher(args_cli)          # ← Omniverse Kit 启动,pxr 此时被注入
simulation_app = app_launcher.app

# All IsaacLab dependent modules should be imported after the app is launched
import sim_evals.environments                 # ← pxr 已可用
```

脚本外**裸 import `isaaclab.sim`** 会报 `ModuleNotFoundError: No module named 'pxr'`——**不是 env 坏了**,只是没走 AppLauncher。别被这个 dry-run 式测试误导。

### 阶段 D:Runtime errors

#### D1. sim-evals 场景资产没下载

症状:
```
Failed to open layer @.../assets/scene1.usd@
```

sim-evals README 明写要单独从 HF 下载:
```
cd /home/zuxinrui/sim-evals
huggingface-cli download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
```

#### D2. 机器人 USD 文件名不匹配

`scene1.usd` 里的 payload 引用 `my_droid.usdz`(写死),但 HF 数据集现在提供的是 `franka_robotiq_2f_85_flattened.usd`。上游 repo 和 HF 数据集版本漂移。

**修复**(symlink,永久留在 assets 里):
```
cd /home/zuxinrui/sim-evals/assets && ln -s franka_robotiq_2f_85_flattened.usd my_droid.usdz
```

USD 靠 magic bytes 判格式,扩展名不严格 —— `.usdz` 指向 `.usd` 文件能正确解析。

#### D3. websockets 版本太老

```
TypeError: connect() got an unexpected keyword argument 'ping_interval'
```

dreamzero 的 `policy_client.py:55` 调 `websockets.sync.client.connect(ping_interval=...)`,需要 **websockets 13+**。env_isaaclab_legacy 默认装的是 12.0(被 `isaacsim-kernel` pin 住)。

```
pip install --upgrade "websockets>=13"
```

**冲突警告可忽略**:`isaacsim-kernel 5.1.0.0 requires websockets==12.0`,但 Isaac Sim 自己只在 livestream/Web UI 用 websockets,headless/desktop 模式下不碰,16.0 版本兼容得很好。

#### D4. `cv2.imshow` 需要 GUI backend

```
cv2.error: The function is not implemented. Rebuild the library with ... GTK+ 2.x or Cocoa support
```

env 里装的是 `opencv-python-headless`(Isaac Sim 带的),不带 GTK/Qt。dreamzero 的 `run_sim_eval.py:199` 想弹个 "Right Camera" 预览窗口,崩。

**修复**:注释掉那两行(`pass`),本地 patch `dreamzero/eval_utils/run_sim_eval.py`:
```python
if not headless:
    # cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)
    pass
```

Isaac Sim 自己的 viewport 已经显示全场景,cv2 那个小窗口纯冗余。

想恢复:`cd /home/zuxinrui/dreamzero && git checkout -- eval_utils/run_sim_eval.py`。

**别装 `opencv-python`**:它和 `opencv-python-headless` 共享 `cv2` namespace,二选一,Isaac Sim 依赖 headless 版本。

### 阶段 E:系统与图形栈

#### E1. inotify watcher limit 超限(报错误导人)

```
[Error] [carb] Failed to create change watch for ...: errno=28/No space left on device
```

**不是磁盘满,不是内存不够**。是 Linux 内核的 `inotify` watcher 数量上限(默认 8192 或 65536)被 Isaac Sim GUI 的几千个 extension 吃满了。Headless 模式不加载 GUI,不爆。

**永久修复**:
```
echo "fs.inotify.max_user_watches=524288" | sudo tee /etc/sysctl.d/99-inotify.conf && sudo sysctl --system
```

#### E2. Vulkan swapchain 建不起来(双卡 display 关键坑)

```
[Error] [carb.graphics-vulkan.plugin] Failed to find a graphics and/or presenting queue.
[Error] [carb.graphics-vulkan.plugin] - GPU 'NVIDIA GeForce RTX 3090' cannot present rendered content from the window to the screen.
[Error] [omni.kit.renderer.plugin] createSwapchain failed.
```

**原因**:双 3090 中**显示器只接了 GPU0**。Vulkan swapchain(render → window)必须在接显示器的 GPU 上建。你用 `CUDA_VISIBLE_DEVICES=1` 强指 GPU1(纯算卡,无 display 输出) → swapchain 失败。

Headless 模式不需要 swapchain(render 到 memory),GPU1 完全够用。**一开 `--no-headless`**,必须切 GPU0。

**修复**:两个 terminal 的 `CUDA_VISIBLE_DEVICES` **互换**:
- policy server → GPU1
- sim client(GUI)→ GPU0

两张 3090 完全等价,哪个跑哪个功能都行,只是 display 通路强制了 sim 必须 GPU0。

**验证显示器接在哪张**:
```
nvidia-smi -q -d UTILIZATION | grep -E "Minor|Display"
# Display Active: Enabled 的就是接显示器的
```

附带消失的错误:`advanceCurrentFrame: backbuffers are not initialized!`——swapchain 建成后不再刷屏。

## 6. 修改清单(本机全部非默认改动)

用于灾后恢复或换机复现。

### openpi
- **分支** `franka-scripts`(基于 main)
  - `scripts/main.py`, `test*.py`, `readme.md`, `third_party/panda_python-*.whl`(从 openpi_franka port)
  - `CLAUDE.md`(快速参考)
  - 本文档 `docs/DROID_ISAAC_SIM_DEPLOYMENT.md`
- `.gitignore` 加 `runs/`(可选,避免 sim 输出进提交)
- `.venv/` 由 `uv sync` 生成(不进 git)

### dreamzero
- **本地 patch**(未提交):`eval_utils/run_sim_eval.py:198-202` 注释掉 cv2.imshow
- `CLAUDE.md` 指向本文档

### sim-evals
- HF assets 下载到 `/home/zuxinrui/sim-evals/assets/`
- **Symlink** `assets/my_droid.usdz` → `franka_robotiq_2f_85_flattened.usd`

### env_isaaclab_legacy 额外装的包
```
pip install tyro mediapy
pip install --upgrade "websockets>=13"
pip install -e /home/zuxinrui/openpi/packages/openpi-client --no-deps
pip install -e /home/zuxinrui/sim-evals --no-deps
pip install "numpy==2.3.1" --force-reinstall  # 只在 isaaclab3 被污染时用
```

### 系统层
```
/etc/sysctl.d/99-inotify.conf:
  fs.inotify.max_user_watches=524288
```

## 7. 排障速查表

| 症状 | 多半是什么 | 哪节 |
|---|---|---|
| `Command 'uv' not found` | uv 没装 | A1 |
| `numpy` 被降 / Isaac Sim 挂 | pip 解析 openpi-client deps | A3 |
| `Config 'X' not found. Did you mean 'X_polaris'?` | 上游 config 改名 | B1 |
| `KeyError: 'observation/gripper_position'` | simple_client 默认 ALOHA | B3 |
| `AttributeError: No isaaclab.envs.mdp attribute observations` | Isaac Lab 版本太新 | C1 |
| `ModuleNotFoundError: No module named 'pxr'` | AppLauncher 没先启动 | C2 |
| `Failed to open layer @.../scene1.usd@` | assets 没下 | D1 |
| `Could not open asset @.../my_droid.usdz@` | symlink 缺失 | D2 |
| `unexpected keyword argument 'ping_interval'` | websockets < 13 | D3 |
| `cv2.error ... GTK+ ... support` | opencv-headless 无 GUI | D4 |
| `errno=28/No space left` 刷屏 | inotify 上限 | E1 |
| `Failed to find graphics and/or presenting queue` | GPU 没接显示器 | E2 |
| `backbuffers are not initialized!` 刷屏 | E2 的后果 | E2 |

## 8. 参考链接

- openpi: https://github.com/Physical-Intelligence/openpi
- dreamzero: https://github.com/zuxinrui/dreamzero(上游:NVIDIA GEAR Lab)
- sim-evals: https://github.com/arhanjain/sim-evals
- sim assets: https://huggingface.co/datasets/owhan/DROID-sim-environments
- openpi polaris config: `src/openpi/training/misc/polaris_config.py`
- openpi DROID obs schema: `src/openpi/policies/droid_policy.py:11-17`
- dreamzero sim client: `eval_utils/run_sim_eval.py`
- dreamzero WebSocket wrapper: `eval_utils/policy_client.py`

## 9. 下一步可做

- 跑多 episode batch eval 统计成功率:`--episodes 50 --headless`
- 切 scene 2 / 3 看 π0.5 泛化能力
- 换 `pi0_fast_droid_jointpos_polaris` 对比 autoregressive vs flow
- LoRA 微调 openpi 策略:`uv run scripts/train.py pi0_libero_low_mem_finetune ...`(不同 task)
- 把这套 pipeline 迁到真机 Franka:用 `scripts/main.py`(`franka-scripts` 分支)替代 `run_sim_eval.py`
