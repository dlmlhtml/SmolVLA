"""Small-data SmolVLA training on LIBERO episode 0, with save/reload verification."""

import argparse
import gc
import json
import time
from pathlib import Path

import draccus
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from lerobot.utils.feature_utils import dataset_to_policy_features


def load_config(path, device):
    # 直接解码配置，绕过 Python 3.14 下 draccus 命令行解析的兼容问题。
    with open(path) as f:
        data = json.load(f)
    assert data.pop("type") == "smolvla"
    data["device"] = device
    return draccus.decode(SmolVLAConfig, data)


def main():
    # 1. 训练参数：steps 是参数更新次数；相对输出路径以启动命令时的目录为准。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs") / time.strftime("smolvla_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("steps, batch-size and lr must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model_id = "lerobot/smolvla_base"
    config = load_config(hf_hub_download(model_id, "config.json"), args.device)


    # 2. 根据数据集对齐输入输出：两路图像、8 维状态、7 维动作。
    base = LeRobotDataset("lerobot/libero", episodes=[0], video_backend="pyav")
    features = dataset_to_policy_features(base.meta.features)
    config.input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    config.output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    # 冻结视觉语言主干，训练动作专家及相关投影层，包括状态投影。
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    config.push_to_hub = False

    # 每个当前观测配上从当前时刻开始的 50 步示范动作。
    # 越过 episode 末尾的位置由 action_is_pad 标记，模型算 loss 时会排除。
    dataset = LeRobotDataset(
        "lerobot/libero", episodes=[0], video_backend="pyav",
        delta_timestamps={"action": [i / base.meta.fps for i in range(config.chunk_size)]},
    )
    # 预处理编码任务文字并归一化状态/动作；后处理把预测动作还原到数据集尺度。
    preprocess, postprocess = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # 固定训练样本用于前后对比，不是独立验证集。
    probe_raw = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    task = probe_raw["task"][0]
    print("task:", task, flush=True)
    print("device:", args.device, "steps:", args.steps, "frames:", len(dataset), flush=True)
    
    probe_batch = preprocess(probe_raw)
    print("INPUTS:", flush=True)
    for key, value in probe_batch.items():
        if isinstance(value, torch.Tensor):
            print(key, tuple(value.shape), value.dtype, flush=True)

    # 3. 加载预训练权重，优化器只接收 requires_grad=True 的参数。
    policy = SmolVLAPolicy.from_pretrained(model_id, config=config, strict=True).to(args.device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    expert_probe = next(p for p in policy.model.vlm_with_expert.lm_expert.parameters() if p.requires_grad)
    initial_parameter = expert_probe.detach().clone()

    # 固定诊断用的噪声和混合时间，避免随机噪声干扰前后 loss 对比。
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, device=args.device)
    fixed_time = torch.full((1,), 0.5, device=args.device)

    def probe_loss(model, batch):
        model.eval()
        with torch.no_grad():
            loss, _ = model(batch, noise=noise, time=fixed_time)
        assert torch.isfinite(loss)
        return loss.item()

    before_loss = probe_loss(policy, probe_batch)
    print("fixed training-sample loss before:", before_loss, flush=True)
    iterator = iter(loader)
    started = time.monotonic()

    # 4. 训练循环：每次取一个 batch，计算 loss，再更新一次参数。
    with open(args.output / "metrics.jsonl", "w") as log:
        for step in range(1, args.steps + 1):
            try:
                raw = next(iterator)
            except StopIteration:
                # 数据遍历完就开始下一轮，直到完成指定的更新次数。
                iterator = iter(loader)
                raw = next(iterator)
            batch = preprocess(raw)
            policy.train()

            optimizer.zero_grad(set_to_none=True)  # 清除上一轮梯度，防止意外累加。
            loss, _ = policy(batch)  # 调用 forward，计算 flow-matching 速度预测误差。
            assert torch.isfinite(loss), "Non-finite loss"
            loss.backward()  # 计算可训练参数的梯度，此时还没有更新权重。

            # 限制总梯度范数；返回值 norm 是裁剪之前的范数。
            norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0, error_if_nonfinite=True)
            assert all(p.grad is None for p in policy.model.vlm_with_expert.vlm.parameters())

            optimizer.step()  # AdamW 根据梯度真正更新权重。
            row = {"step": step, "loss": loss.item(), "gradient_norm_before_clip": norm.item(),
                   "elapsed_seconds": time.monotonic() - started}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"step {step}/{args.steps} loss={row['loss']:.6f} grad_norm={norm.item():.4f}", flush=True)


    # 5. 检查专家权重发生变化，并保存权重、配置和归一化处理器。
    change = (expert_probe.detach() - initial_parameter).abs().max().item()
    assert change > 0, "Expert weights did not change"
    after_loss = probe_loss(policy, probe_batch)
    checkpoint = args.output / "checkpoint"
    policy.save_pretrained(checkpoint)
    preprocess.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    postprocess.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    print("checkpoint saved:", checkpoint, flush=True)


    # 6. 释放旧模型和优化器，独立重载 checkpoint，核对同一条件下的 loss。
    del optimizer, trainable, expert_probe, initial_parameter, policy, loss
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    restored_config = load_config(checkpoint / "config.json", args.device)

    restored = SmolVLAPolicy.from_pretrained(checkpoint, config=restored_config, strict=True)
    restored_pre, restored_post = make_pre_post_processors(restored_config, str(checkpoint))
    restored_batch = restored_pre(probe_raw)
    restored_loss = probe_loss(restored, restored_batch)
    assert abs(restored_loss - after_loss) < 1e-5, "Reload changed fixed-probe loss"

    # 推理不提供示范答案，只输入图像、状态和语言。
    observation = dict(restored_batch)
    observation.pop("action", None)
    observation.pop("action_is_pad", None)
    with torch.inference_mode():

        chunk = restored.predict_action_chunk(observation)
        actions = restored_post(chunk)  # 反归一化后得到 [1, 50, 7]，尚未在环境中执行。

    assert actions.shape == (1, config.chunk_size, 7) and torch.isfinite(actions).all()
    with open(args.output / "predicted_actions.json", "w") as f:
        json.dump(actions.cpu().tolist(), f)
    report = {
        "task": task, "dataset": "lerobot/libero", "episodes": [0], "fps": dataset.meta.fps,
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
        "seed": args.seed, "device": args.device,
        "fixed_training_sample_loss_before": before_loss,
        "fixed_training_sample_loss_after": after_loss, "reloaded_loss": restored_loss,
        "expert_parameter_max_change": change, "output_shape": list(actions.shape),
        "evaluation_scope": "Training-sample diagnostic only; no held-out evaluation or rollout.",
        "normalization": "Dataset-provided statistics; saved with processors.",
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print("fixed training-sample loss:", before_loss, "->", after_loss, flush=True)
    print("reloaded action shape:", actions.shape, flush=True)
    print("TRAIN / SAVE / RELOAD / INFERENCE PASSED", flush=True)


if __name__ == "__main__":
    main()
