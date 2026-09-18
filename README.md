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

## 4060 新环境：从这里开始

要求 Linux 或 Windows WSL2，已安装 NVIDIA 驱动、Git、Python 3.12 及 venv 支持。先在该系统中确认 `nvidia-smi` 能看到 4060。

```bash
git clone https://github.com/dlmlhtml/SmolVLA.git
cd SmolVLA
bash setup_4060.sh
source .venv/bin/activate
```

安装脚本固定 LeRobot 源码到本次实验使用的 commit：`b6ec0060779550c0a157ae34feb89e0cf86012a8`。依赖按该版本约束解析，并在安装后输出 `environment-installed.txt`；这不是完整依赖锁文件。4060 的实际显存占用尚未测量。

## 迁移已经训练的模型

模型约 865 MB，单独打包为 `smolvla-100steps-checkpoint.tar.gz`，不放进 Git 历史。该文件需要另行下载或复制；若已发布 Release，从 Release 获取。当前仓库内容本身不含模型权重。

把压缩包放到本项目目录，验证随附的 SHA256 后解压：

```bash
# 发布后可从此 Release 下载模型；也可通过网页下载同名附件。
gh release download checkpoint-100steps --repo dlmlhtml/SmolVLA \
  --pattern 'smolvla-100steps-checkpoint.tar.gz*'
sha256sum -c smolvla-100steps-checkpoint.tar.gz.sha256
tar -xzf smolvla-100steps-checkpoint.tar.gz
```

应得到 `outputs/smolvla_20260914_233324/summary.json` 及完整 `checkpoint/`。必须保留处理器统计量和 tokenizer，不要只拷贝 model.safetensors。

## 录制模型实际执行的视频

```bash
MUJOCO_GL=egl python rollout_smolvla.py \
  --checkpoint outputs/smolvla_20260914_233324/checkpoint \
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
  --checkpoint outputs/smolvla_20260914_233324/checkpoint
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
