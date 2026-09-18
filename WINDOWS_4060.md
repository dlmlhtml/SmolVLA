# Windows RTX 4060：从安装到训练视频

保留 Windows，通过 WSL2 运行 Ubuntu 24.04，使用 Windows 上的 4060。无需双系统。
这是本项目完整流程的目标环境；本仓库依赖的 LIBERO extra 限定 Linux，因此这里不提供原生 Windows 的完整 LIBERO 运行承诺。
当前尚未在你的 Windows 机器实测，显存、EGL 渲染及动作对齐仍需验证。

## 1. Windows 管理员 PowerShell：安装 WSL2

先安装/更新 Windows NVIDIA 显卡驱动，在 PowerShell 中确认 `nvidia-smi` 能看到 RTX 4060。
WSL 使用 Windows 提供的 GPU 驱动，不要在 Ubuntu 里另装 NVIDIA Linux 显卡驱动。

在管理员 PowerShell 执行：

```powershell
wsl --install -d Ubuntu-24.04
```

也可以在已下载仓库的目录执行提供的安装入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

按安装提示重启 Windows，首次启动 Ubuntu 时创建 Linux 用户名和密码。
如果已经安装 Ubuntu-24.04，不必重新安装；先检查版本：

```powershell
wsl --update
wsl --list --verbose
# 仅当该发行版的 VERSION 显示 1 时执行下面的转换
wsl --set-version Ubuntu-24.04 2
wsl -d Ubuntu-24.04
```

以下命令全部在 Ubuntu 终端运行，不要粘贴到 PowerShell。

## 2. Ubuntu：安装系统依赖和项目

```bash
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv python3.12-dev \
  build-essential pkg-config ffmpeg libegl1 libgl1 libglfw3 libosmesa6

mkdir -p ~/projects
cd ~/projects
git clone https://github.com/dlmlhtml/SmolVLA.git
cd SmolVLA
bash setup_4060.sh
source .venv/bin/activate
```

项目放在 Ubuntu 的 `~/projects`，不复用 Windows 的 Python 或虚拟环境。
安装脚本检查 GPU、固定 LeRobot 源码版本，并安装训练和 LIBERO 依赖。
如果提示 CUDA 不可用，先确认 WSL2 中的 `nvidia-smi`；若命令不在 PATH，可试 `/usr/lib/wsl/lib/nvidia-smi`。

## 3. Ubuntu：重新训练和评估

```bash
cd ~/projects/SmolVLA
source .venv/bin/activate
python train_smolvla.py --steps 100 --batch-size 1 --device cuda --output outputs/smolvla_4060
python eval_smolvla_offline.py --device cuda --checkpoint outputs/smolvla_4060/checkpoint
```

首次下载基础模型和数据需要网络。输出目录必须是新目录，再次训练时换一个名称。
若发生显存不足，请保留完整报错；当前尚未测量你的 4060 配置下训练峰值，不保证 batch size 1 一定能完成训练。

## 4. Ubuntu：先检查渲染，再录完整视频

先用 5 步检查环境、模型加载和 MP4 编码：

```bash
MUJOCO_GL=egl python rollout_smolvla.py --device cuda \
  --checkpoint outputs/smolvla_4060/checkpoint --max-steps 5
```

成功后运行完整轨迹：

```bash
MUJOCO_GL=egl python rollout_smolvla.py --device cuda \
  --checkpoint outputs/smolvla_4060/checkpoint \
  --episodes 1 --max-steps 520 --action-steps 10
```

如果 WSL 的 EGL 初始化失败，可以试软件渲染：将命令中的 `MUJOCO_GL=egl` 换为 `MUJOCO_GL=osmesa`。
软件渲染较慢；这只切换 MuJoCo 渲染后端，模型仍使用 `--device cuda`。
相机方向、控制频率、动作语义尚需核对，见 SMOLVLA_ROLLOUT.md。

## 5. 在 Windows 打开视频

在 Ubuntu 终端执行：

```bash
cd ~/projects/SmolVLA/outputs/smolvla_4060
explorer.exe .
```

在资源管理器打开最新的 `rollout_时间` 文件夹，播放 `episode_000.mp4`。
`results.json` 记录是否成功；视频可能展示失败，离线 loss 下降不保证任务完成。

## 官方依据

- [Microsoft：安装 WSL](https://learn.microsoft.com/windows/wsl/install)
- [Microsoft：WSL2 中使用 CUDA](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl)
- [NVIDIA：CUDA on WSL，使用 Windows 驱动](https://docs.nvidia.com/cuda/wsl-user-guide/)
