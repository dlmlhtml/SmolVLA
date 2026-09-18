# SmolVLA 小数据训练任务

任务：使用 `lerobot/libero` 的 episode 0，学习“把白色杯子放在左边盘子上，把黄白色杯子放在右边盘子上”的示范动作。
这是单 episode 的行为拟合实验，不是已经通过仿真成功率评估的策略。

## 运行

```bash
python3 train_smolvla.py --steps 100
```

默认 batch size 为 1，学习率为 1e-5，随机种子为 42；有 CUDA 时自动使用 CUDA，否则使用 CPU。
`--steps 5` 可验证完整流程。每个 step 是一次优化器更新，不是一个 episode，也不是一个 epoch。
每次启动从基础 checkpoint 开始，默认创建独立的输出目录。脚本不提供断点续训。

## 完整流程

1. 读取基础 checkpoint 配置，并依据数据集设置图片、状态、动作字段。
2. 从 episode 0 的 214 帧中随机采样，以每个起始帧构造当前观测和 50 步动作目标。
3. 使用数据集统计量归一化状态和动作，把任务文字编码为 tokens。
4. 冻结 VLM，训练动作专家、状态投影、动作投影和时间处理层。
5. 循环计算 flow-matching loss、反向传播、梯度裁剪和 AdamW 更新。
6. 保存模型权重、配置和带归一化统计量的输入输出处理器。
7. 独立重新加载模型和处理器，检查固定样本 loss 一致，再生成并反归一化动作序列。

## 输入输出

`B` 是 batch size。图片是当前观测，不是未来 50 张图片。

| 字段 | 预处理后形状 | 含义 |
| --- | --- | --- |
| observation.images.image | [B, 3, 256, 256] | 第一路当前 RGB 图像 |
| observation.images.image2 | [B, 3, 256, 256] | 第二路当前 RGB 图像 |
| observation.state | [B, 8] | 当前机器人状态，按数据集统计量归一化 |
| observation.language.tokens | [B, 48] | 任务文字的 token ID |
| observation.language.attention_mask | [B, 48] | 有效语言位置 |
| action | [B, 50, 7] | 从当前时刻开始的示范动作序列，训练目标 |
| action_is_pad | [B, 50] | episode 边界外的填充位置，不计入 loss |

episode/frame/timestamp 等字段用于定位数据，不是动作专家的主要条件输入。
模型内部会进一步处理图像，并把状态和动作补齐到配置中的最大维度。

训练时：`policy(batch)` 返回 `(loss, metrics)`。`loss` 是可反向传播的标量，模型学习带噪动作对应的速度目标。
推理时：去掉示范 action，`predict_action_chunk()` 输出 `[1, 50, 7]`，再由 postprocessor 反归一化。
7 维动作遵循 LIBERO 数据定义；本实验没有连接环境执行。

## 生成文件

每次运行生成 `outputs/smolvla_时间/`：

- `metrics.jsonl`：每一步的 loss、裁剪前梯度范数、累计训练时间。
- `checkpoint/`：模型权重、模型配置、输入输出处理器配置与统计量。
- `predicted_actions.json`：重新加载后预测的完整动作序列，形状 [1, 50, 7]，已反归一化。
- `summary.json`：任务、超参数、固定训练样本的前后 loss、重载验证与输出形状。

固定样本比较使用相同 noise 和 time，便于观察参数更新带来的变化。
该样本来自训练 episode，因此这里只是训练诊断，没有独立验证集；短跑 loss 不保证下降。
泛化表现需要后续独立 episode 和环境 rollout 评估，不能由这些 loss 推断成功率。

## 离线评估

```bash
python3 eval_smolvla_offline.py --checkpoint outputs/smolvla_20260914_233324/checkpoint
```

脚本从训练报告读取任务与训练 episode，通过 Parquet 的 task_index 自动寻找同任务、未用于本次微调的 episode。
默认每段 episode 等距抽取 3 帧；可通过 `--samples` 增加样本数。
对基础模型和微调模型使用相同的 7 维配置、已保存的归一化处理器、固定噪声和 t=0.5。

- flow loss：给定带噪示范动作时的速度预测误差，越低越好。
- normalized action MSE：从噪声实际生成动作序列后，与示范动作比较的归一化均方误差，越低越接近该示范。
- 两项指标均排除 episode 边界填充，汇总按有效动作数量加权。

结果保存在训练目录下的 `offline_eval_时间/`，包含 `RESULTS.md`、`summary.json` 和逐样本 `samples.jsonl`。
这是少量样本的初步诊断；一段新 episode 不能支持广泛泛化结论。
归一化沿用数据集全局统计量，且无法确定基础预训练是否见过这些数据，因此不能称为严格独立测试集。
动作误差不等于环境成功率，当前脚本不执行机器人动作。
