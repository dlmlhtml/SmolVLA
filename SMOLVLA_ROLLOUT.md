# SmolVLA 真实执行视频

`rollout_smolvla.py` 在 LIBERO 环境中加载已训练模型，实时读观测、执行动作并录制 MP4。
这不是训练数据回放，也不是把 predicted_actions.json 直接播成视频。

## 当前验证状态

当前 Mac 缺少 libero、robosuite 和 mujoco，仓库的 LIBERO extra 限定 Linux。
脚本已做语法和环境检查，尚未执行真实仿真；当前没有生成 rollout 视频。
需要在带 NVIDIA 驱动的 Linux / WSL2 环境继续安装、实际测试渲染与录制。

## Linux / WSL2 准备

按 README 安装固定版本的 LeRobot，并在该机器重新训练生成 outputs/smolvla_4060。
训练输出目录必须包含 summary.json 和完整 checkpoint/（包括处理器和 tokenizer）。
在 Python 3.12 的独立环境中，从仓库根目录安装：

```bash
python -m pip install -e '.[smolvla,libero]'
python rollout_smolvla.py --check
```

依赖安装、LIBERO 资源和 EGL 渲染还需在目标机器验证；当前没有验证 4060 的显存峰值。

## 录制

把下面路径换成训练输出在 Linux 上的位置：

```bash
MUJOCO_GL=egl python rollout_smolvla.py \
  --checkpoint /path/to/smolvla_20260914_233324/checkpoint \
  --episodes 1 --max-steps 520 --action-steps 10
```

脚本按 summary.json 的任务文字定位套件和 task_id，并使用官方初始状态。
每次预测完整 chunk，但默认执行前 10 步后根据新观测重新预测。
输出位于原训练目录的 rollout_时间/，包含：

- episode_000.mp4：主视角和腕部视角并排显示的实际执行画面。
- actions_000.jsonl：每步预测动作、实际执行动作、奖励和成功标记。
- results.json：任务、参数、是否成功、步数、裁剪次数和错误信息。

视频按仿真控制频率编码，播放长度不包含模型在 CPU/GPU 上计算时的等待时间。
代码将反归一化动作限制在环境 action_space 范围内，并记录裁剪次数。

## 对齐与解释

默认采用官方 LiberoProcessorStep 的 180 度图像旋转和 8 维状态转换，控制模式为相对控制，频率为 20 Hz。
本训练数据 lerobot/libero 的卡片仅给出采样频率 10 FPS，没有说明是否下采样及相机转换来源；不能由 FPS 断定控制频率。
因此图像方向、动作语义及控制时序仍需要在目标运行环境结合数据核对。
`--image-rotation 0` 和 `--control-freq` 用于显式调整和记录对齐设置，不能以任务是否成功为由盲调。

数据卡：https://huggingface.co/datasets/lerobot/libero

一次成功或失败都只是一次 rollout 结果。单 episode 微调 100 步并不保证完成摆杯子任务。
