# SmolVLA：100 步微调 checkpoint 与完整学习流程

本项目包含从预训练推理、单步训练验证、多步训练、保存重载，到同任务离线评估和待验证的仿真录制代码。

## 已完成的实验

- 模型：lerobot/smolvla_base。
- 数据：lerobot/libero，episode 0，摆放两只杯子的任务。
- 训练：100 次更新，batch size 1，学习率 1e-5，Mac CPU。
- 离线评估：训练 episode 0 与同任务 episode 18 各取 3 帧；基础权重和微调权重使用相同观测、归一化和噪声。
- episode 18 的归一化动作 MSE 从 1.073904 降至 0.649092。

这些少量离线样本不能证明任务成功率。未运行过真实 LIBERO rollout，4060 / Linux 环境仍需验证。

## 附件

- smolvla-100steps-checkpoint.tar.gz：权重、完整输入输出处理器、tokenizer、训练 summary.json。
- smolvla-100steps-checkpoint.tar.gz.sha256：压缩包完整性校验。

在项目根目录验证并解压后会生成 outputs/smolvla_20260914_233324/。
完整训练日志、预测动作和离线比较报告位于仓库 results/。
模型可直接继续评估和推理；原训练脚本未保存优化器状态，不能声称从第 101 步精确恢复训练。

4060 环境的安装与运行命令见 README.md。
