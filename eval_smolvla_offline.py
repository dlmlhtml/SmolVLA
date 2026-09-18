"""离线比较基础模型与微调模型：使用相同样本、归一化和噪声，不更新参数、不执行环境动作。"""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from torch.utils.data import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from train_smolvla import load_config


def main():
    # 1. 读取评估参数；samples 是每个 episode 抽取的帧数，不是训练步数。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    checkpoint = args.checkpoint.resolve()
    # 训练报告告诉我们模型学过哪些 episode、对应什么任务，避免把训练数据当新数据。
    with open(checkpoint.parent / "summary.json") as f:
        training = json.load(f)
    config = load_config(checkpoint / "config.json", args.device)
    # 复用训练时保存的 tokenizer 和归一化统计量，两种模型都使用这一套处理器。
    preprocess, _ = make_pre_post_processors(
        config, str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    train_episodes = training["episodes"]
    reference = LeRobotDataset(training["dataset"], episodes=[train_episodes[0]], video_backend="pyav")
    task = training["task"]


    # 2. 寻找任务相同、但未参与本次微调的 episode。
    # 元数据没有逐 episode 的任务文字，所以从 Parquet 中读取任务编号来匹配。
    heldout_episode = None
    task_index = int(reference.meta.tasks.loc[task, "task_index"])
    filesystem = HfFileSystem()
    visited = set()
    for ep in reference.meta.episodes:
        candidate_id = int(ep["episode_index"])
        relative_path = reference.meta.get_data_file_path(candidate_id)
        # 多个 episode 可能存在同一个 Parquet 文件中，每个文件只扫描一次。
        if relative_path in visited:
            continue
        visited.add(relative_path)
        local = reference.root / relative_path
        if local.exists():
            table = pq.read_table(local, columns=["episode_index", "task_index"])
        else:
            remote = f"datasets/{training['dataset']}@{reference.revision}/{relative_path}"
            with filesystem.open(remote, "rb") as f:
                table = pq.read_table(f, columns=["episode_index", "task_index"])
        pairs = table.to_pandas().drop_duplicates()
        # 同时满足：任务编号一致，并且 episode 不在训练列表里。
        matches = pairs[(pairs.task_index == task_index) & ~pairs.episode_index.isin(train_episodes)]
        if not matches.empty:
            heldout_episode = int(matches.iloc[0].episode_index)
            break
    if heldout_episode is None:
        raise ValueError("No held-out episode found")
    output = checkpoint.parent / time.strftime("offline_eval_%Y%m%d_%H%M%S")
    output.mkdir(exist_ok=False)
    print("task:", task, flush=True)
    print("train episode:", train_episodes[0], "held-out episode:", heldout_episode, flush=True)
    samples = []


    # 3. 分别准备训练段和新轨迹的样本；两组都只是抽样评估，并未遍历所有帧。
    for split, episode in (("train", train_episodes[0]), ("heldout", heldout_episode)):
        dataset = LeRobotDataset(
            training["dataset"], episodes=[episode], video_backend="pyav",
            force_cache_sync=(split == "heldout"),
            # 当前观测对应从当前时刻开始的 50 步示范动作，作为比较目标。
            delta_timestamps={"action": [i / reference.meta.fps for i in range(config.chunk_size)]},
        )
        # 默认等距取起点、中段、末帧；末帧之后的动作由 action_is_pad 标记为填充。
        indices = torch.linspace(0, len(dataset) - 1, min(args.samples, len(dataset))).round().int().tolist()
        for index in indices:
            raw = dataset[index]
            assert raw["task"] == task and raw["episode_index"].item() == episode
            # default_collate 添加 batch 维度；preprocess 编码文字并归一化状态和动作。
            batch = preprocess(default_collate([raw]))
            samples.append({"split": split, "episode": episode, "frame": raw["frame_index"].item(),
                            "batch": batch})
        print(split, "frames:", indices, flush=True)

    records = []


    # 4. 依次评估两套权重，避免同时把两个模型放进内存。
    with open(output / "samples.jsonl", "w") as log:
        for label, source in (("base", "lerobot/smolvla_base"), ("finetuned", str(checkpoint))):
            # 基础权重也使用相同的 7 维动作配置；deepcopy 避免共享配置被意外修改。
            model = SmolVLAPolicy.from_pretrained(source, config=copy.deepcopy(config), strict=True)
            model.eval()
            # 只计算指标，没有 backward 或 optimizer.step，不会继续训练。
            with torch.inference_mode():
                for sample_id, sample in enumerate(samples):
                    batch = copy.deepcopy(sample["batch"])
                    # 同一 sample_id 在两套模型下使用同一种子，保证噪声完全一致。
                    generator = torch.Generator(device=args.device).manual_seed(args.seed + sample_id)
                    noise = torch.randn((1, config.chunk_size, config.max_action_dim),
                                        generator=generator, device=args.device)
                    fixed_time = torch.full((1,), 0.5, device=args.device)
                    # 指标一：forward 在固定混合时间 t=0.5 上计算 flow-matching 速度误差。
                    # model(batch) 调用 forward；返回 (loss, 指标字典)，这里仅保留标量 loss。
                    loss, _ = model(batch, noise=noise.clone(), time=fixed_time)
                    # 指标二：去掉示范答案，模型仅凭观测和噪声真正生成动作序列。
                    observation = {k: v for k, v in batch.items() if k not in ("action", "action_is_pad")}
                    # 各帧是独立评估样本，清空之前样本的队列状态。
                    model.reset()
                    prediction = model.predict_action_chunk(observation, noise=noise.clone())
                    target = batch["action"]
                    assert prediction.shape == target.shape
                    assert torch.isfinite(loss) and torch.isfinite(prediction).all()
                    # ~ 对布尔 mask 取反：只比较 episode 内真实存在的动作。
                    # prediction 和 target 都在归一化空间，所以不需要反归一化再算此指标。
                    valid = ~batch["action_is_pad"]
                    error = (prediction - target).square()[valid]
                    # error 形状为 [有效时间步数, 7]；mean() 得到动作均方误差 MSE。
                    record = {"model": label, "split": sample["split"], "episode": sample["episode"],
                              "frame": sample["frame"], "seed": args.seed + sample_id,
                              "flow_loss_t05": loss.item(), "valid_action_steps": int(valid.sum().item()),
                              "normalized_action_mse": error.mean().item(),
                              "squared_error_sum": error.sum().item(), "action_elements": error.numel(),
                              "normalized_mse_per_dimension": error.mean(dim=0).cpu().tolist()}
                    records.append(record)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"{label} {sample['split']} frame={sample['frame']} "
                          f"loss={loss.item():.5f} action_mse={error.mean().item():.5f}", flush=True)
            del model
            # 释放当前模型，再加载下一套权重；CUDA 缓存仅在使用 GPU 时清理。
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    aggregate = []

    
    # 5. 汇总为基础/微调 × 训练段/新轨迹四组结果。
    # 按有效动作数量加权，避免只有 1 步有效动作的末帧和完整 50 步样本占相同权重。
    for label in ("base", "finetuned"):
        for split in ("train", "heldout"):
            rows = [r for r in records if r["model"] == label and r["split"] == split]
            weight = sum(r["valid_action_steps"] for r in rows)
            aggregate.append({"model": label, "split": split, "samples": len(rows),
                              "flow_loss_t05": sum(r["flow_loss_t05"] * r["valid_action_steps"] for r in rows) / weight,
                              "normalized_action_mse": sum(r["squared_error_sum"] for r in rows) /
                              sum(r["action_elements"] for r in rows)})
    limitations = [
        "Small deterministic frame sample from one episode per split, one noise draw and t=0.5.",
        "Held out from this fine-tuning run only; base pretraining overlap is not established.",
        "Both models reuse saved dataset-global normalization statistics, not train-only statistics.",
        "Action MSE measures agreement with one demonstration, not task success. No environment rollout.",
        "Base weights are evaluated with the same adapted 7D configuration as the fine-tuned weights.",
    ]
    report = {"checkpoint": str(checkpoint), "task": task, "train_episodes": train_episodes,
              "heldout_episode": heldout_episode, "seed": args.seed, "device": args.device,
              "aggregate": aggregate, "limitations": limitations}
    # 6. samples.jsonl 保留逐样本结果；summary.json 保存汇总；RESULTS.md 提供可读表格。
    # heldout 只表示未用于本次微调；这些离线误差不能直接换算成机器人任务成功率。
    with open(output / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    lines = ["# Offline evaluation", "", f"Task: {task}", "",
             f"Training episode: {train_episodes[0]}; held-out episode: {heldout_episode}.", "",
             "Both metrics are lower-is-better; padding is excluded and means are weighted by valid actions.", "",
             "| Model | Split | Samples | Flow loss (t=0.5) | Normalized action MSE |",
             "| --- | --- | --- | --- | --- |"]
    for row in aggregate:
        lines.append(f"| {row['model']} | {row['split']} | {row['samples']} | "
                     f"{row['flow_loss_t05']:.6f} | {row['normalized_action_mse']:.6f} |")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in limitations]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("EVALUATION COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
