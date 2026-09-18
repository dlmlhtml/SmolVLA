# SmolVLA 学习与实验

基于 LeRobot 的最小完整实验：预训练推理 → 单 episode 微调 → 离线评估 → LIBERO rollout 录制脚本。

**已验证：** Mac CPU 上完成 100 步微调、保存与重载、同任务新 episode 的离线抽样评估。
**待验证：** RTX 4060 Linux/WSL2 环境安装、仿真渲染、模型闭环执行和视频录制。当前没有机器人任务成功率结果或 rollout 视频。

## 实验任务与结果

任务：把白色杯子放到左边盘子，把黄白色杯子放到右边盘子。

- 基础模型：`lerobot/smolvla_base`。
- 数据：`lerobot/libero`；微调用 episode 0（214 帧），离线比较另用同任务 episode 18。
- 训练：100 次更新、batch size 1、AdamW 学习率 1e-5；冻结 VLM，更新动作专家及相关投影层。
- 输入：两路 RGB 图像、8 维状态、语言指令；输出：50 步 × 7 维动作。

| 数据 | 指标（越低越好） | 基础模型 | 微调模型 |
| --- | --- | ---: | ---: |
| episode 0 | Flow loss | 1.760037 | 0.969049 |
| episode 0 | 归一化动作 MSE | 1.116051 | 0.662704 |
| episode 18 | Flow loss | 1.266111 | 0.939694 |
| episode 18 | 归一化动作 MSE | 1.073904 | 0.649092 |

每段只抽取 3 帧，使用固定噪声及 t=0.5。episode 18 仅保证未用于本次微调；基础模型预训练的数据重叠未知，归一化使用数据集全局统计量。这些结果是初步诊断，不是严格泛化基准，更不等于成功率。

详见 [评估结果](results/offline_eval/RESULTS.md)、[训练记录](results/training_100_steps/summary.json)。

## Windows 4060：从这里开始

**你的 Windows 保持不变，使用 WSL2 + Ubuntu 24.04 跑完整流程。**
请先按 [Windows 4060 安装指南](WINDOWS_4060.md) 完成 WSL、Windows NVIDIA 驱动和 Ubuntu 系统依赖准备。
提供 `setup_windows.ps1` 作为管理员 PowerShell 安装入口；`setup_4060.sh` 则在 Ubuntu 中执行。
完整 LIBERO 流程目前面向 Linux/WSL2，不是把 Bash 命令直接放进 Windows PowerShell。

完成准备后，以下命令在 **Ubuntu 终端** 执行：

```bash
git clone https://github.com/dlmlhtml/SmolVLA.git
cd SmolVLA
bash setup_4060.sh
source .venv/bin/activate
```

安装脚本固定 LeRobot 源码到本次实验使用的 commit：`b6ec0060779550c0a157ae34feb89e0cf86012a8`。依赖按该版本约束解析，并在安装后输出 `environment-installed.txt`；这不是完整依赖锁文件。4060 的实际显存占用尚未测量。

## 在 4060 上重新训练与评估

仓库只保存代码、说明和小体积实验记录，不上传 checkpoint 或模型压缩包。
首次运行会从 Hugging Face 下载基础模型和所需数据，再在本机生成 checkpoint。

```bash
python train_smolvla.py --steps 100 --output outputs/smolvla_4060
python eval_smolvla_offline.py --checkpoint outputs/smolvla_4060/checkpoint
```

后续每次重新打开 Ubuntu，先 `cd ~/projects/SmolVLA` 并 `source .venv/bin/activate`。

训练会保存权重、输入输出处理器、tokenizer、训练日志与 summary.json。
输出目录必须是新目录；再次训练时换一个 --output 路径，避免覆盖之前的实验。

## 录制模型实际执行的视频

```bash
MUJOCO_GL=egl python rollout_smolvla.py \
  --checkpoint outputs/smolvla_4060/checkpoint \
  --episodes 1 --max-steps 520 --action-steps 10
```

预期输出在训练目录下新建的 `rollout_时间/`：

- `episode_000.mp4`：主相机与腕部相机并排显示的真实仿真画面。
- `results.json`：是否成功、步数、运行配置、异常信息。
- `actions_000.jsonl`：每一步预测及实际执行动作。

默认每个 chunk 执行 10 步后重新预测。视频按仿真控制频率播放，不包含模型计算等待时间。该录制脚本目前仅通过语法检查、环境检查及合成观测适配检查，尚未通过真实 rollout。
相机旋转、控制频率及动作语义需要在目标环境核对，详见 [录制说明](SMOLVLA_ROLLOUT.md)。失败的视频同样保留，不把数据集示范视频当作模型执行结果。

## 其他入口

```bash
# 重新训练；默认有 CUDA 就使用 CUDA
python train_smolvla.py --steps 100

# 对已训练模型做同任务离线比较
python eval_smolvla_offline.py \
  --checkpoint outputs/smolvla_4060/checkpoint
```

| 文件 | 内容 |
| --- | --- |
| test_smolvla.py | 基础模型的加载、数据检查和 action chunk 推理 |
| test_smolvla_train_step.py | 单次梯度更新验证 |
| train_smolvla.py | 多步训练、保存、重载和动作输出 |
| eval_smolvla_offline.py | 基础权重与微调权重的配对离线比较 |
| rollout_smolvla.py | 仿真闭环执行与 MP4 录制 |
| SMOLVLA_TRAINING.md | 输入输出及训练流程解释 |

本项目依赖 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)，上游许可证随附于 LICENSE.lerobot；模型与数据的使用请遵循各自仓库条款。
