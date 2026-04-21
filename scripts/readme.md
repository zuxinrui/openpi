network configuration for more details.

## Network configuration

sudo ip link set wlp* down 2>/dev/null || true
sudo ip route del default dev enp3s0 2>/dev/null || true
ip route

sudo ip addr add 172.16.0.1/24 dev enp3s0

ping 172.16.0.2

## Franka Robot Controller configuration

open 172.16.0.2
unlock the joints
activate code execution

## Test the connection

```
python3 scripts/test.py 172.16.0.2
```

## Pi 0.5 Control

```
python3 scripts/main.py
```

## installation details

install librealsense2 with sudo apt install:

https://github.com/IntelRealSense/librealsense/issues/12701

install pyrealsense2:

pip install pyrealsense2

## Extra packages required by scripts/main.py (not in openpi's pyproject.toml)

`scripts/main.py` imports a few packages that upstream openpi does not declare.
Install them manually after `uv sync`:

```
# RealSense Python bindings (camera streaming)
pip install pyrealsense2

# DataFrame helper used for saving trajectories
pip install pandas

# Franka Python bindings (libfranka 0.10.0, Python 3.8 only -- see note below)
pip install third_party/panda_python-0.7.5+libfranka.0.10.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### Python version note

The bundled `panda_python` wheel is built for **Python 3.8** (`cp38` tag), while
openpi itself requires Python 3.11+. Run the policy server (openpi core) in the
Python 3.11 env, and run `scripts/main.py` in a separate Python 3.8 env that
has `panda_py`, `pyrealsense2`, `pandas`, `opencv-python`, and `openpi-client`
installed. The client talks to the server via WebSocket, so the two envs don't
need to share a Python version.

Minimal Python 3.8 client env:

```
conda create -n openpi-franka-client python=3.8 -y
conda activate openpi-franka-client
pip install pyrealsense2 pandas opencv-python tyro tqdm
pip install packages/openpi-client
pip install third_party/panda_python-0.7.5+libfranka.0.10.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

