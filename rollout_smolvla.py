"""Record actual LIBERO execution of a trained SmolVLA checkpoint (Linux runtime)."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def batch_observation(value):
    """官方环境处理器接收带 batch 维度的图像和嵌套机器人状态。"""
    if isinstance(value, dict):
        return {key: batch_observation(item) for key, item in value.items()}
    return value[None, ...] if hasattr(value, "shape") else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check", action="store_true", help="Check runtime without loading model or simulator")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--image-rotation", type=int, choices=[0, 180], default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    # 当前仓库只在 Linux 上声明 LIBERO 支持，避免在 Mac 主环境中强装不兼容依赖。
    required = ["libero", "robosuite", "mujoco", "torch", "av"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    status = {"platform": sys.platform, "missing": missing, "supported_platform": sys.platform == "linux"}
    if args.check:
        print(json.dumps(status, indent=2))
        return
    if sys.platform != "linux" or missing:
        parser.error(f"Linux/WSL2 LIBERO runtime required: {status}. See SMOLVLA_ROLLOUT.md")
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if min(args.episodes, args.max_steps, args.action_steps, args.control_freq) < 1:
        parser.error("episodes, max-steps, action-steps and control-freq must be positive")
    # 无窗口渲染；可以在命令行覆盖为其他 MuJoCo 支持的后端。
    os.environ.setdefault("MUJOCO_GL", "egl")

    import av
    import numpy as np
    import torch
    from libero.libero import benchmark

    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import DataProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep
    from train_smolvla import load_config

    checkpoint = args.checkpoint.resolve()
    with open(checkpoint.parent / "summary.json") as f:
        training = json.load(f)
    task_text = training["task"]
    # 通过训练任务文字查找真实环境任务，不靠猜测 task_id。
    normalize = lambda text: " ".join(text.lower().replace("_", " ").split())
    selected = None
    suites = benchmark.get_benchmark_dict()
    for suite_name in ("libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"):
        if suite_name not in suites:
            continue
        suite = suites[suite_name]()
        for task_id in range(len(suite.tasks)):
            if normalize(suite.get_task(task_id).language) == normalize(task_text):
                selected = (suite_name, suite, task_id)
                break
        if selected:
            break
    if selected is None:
        raise ValueError(f"No exact LIBERO language match for: {task_text}")
    suite_name, suite, task_id = selected
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(checkpoint / "config.json", device)
    if args.action_steps > config.chunk_size:
        parser.error("action-steps must not exceed chunk_size")
    config.n_action_steps = args.action_steps
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        config, str(checkpoint), preprocessor_overrides={"device_processor": {"device": device}},
    )
    # 官方适配器：图像旋转 180 度；位置 + 轴角 + 两个夹爪位置组成 8 维状态。
    env_processor = DataProcessorPipeline(steps=[LiberoProcessorStep()])
    output = checkpoint.parent / time.strftime("rollout_%Y%m%d_%H%M%S")
    output.mkdir(exist_ok=False)
    report = {"checkpoint": str(checkpoint), "task": task_text, "suite": suite_name, "task_id": task_id,
              "device": device, "action_steps": args.action_steps, "control_freq": args.control_freq,
              "image_rotation": args.image_rotation, "seed": args.seed,
              "alignment_status": "Official environment defaults; dataset orientation and control timing need runtime verification.",
              "episodes": []}
    print("task:", task_text, "suite:", suite_name, "task_id:", task_id, flush=True)
    for episode in range(args.episodes):
        torch.manual_seed(args.seed + episode)
        env = LiberoEnv(suite, task_id, suite_name, obs_type="pixels_agent_pos",
                        control_freq=args.control_freq, episode_index=episode, control_mode="relative")
        video_path = output / f"episode_{episode:03d}.mp4"
        result = {"episode": episode, "success": False, "steps": 0, "clipped_components": 0,
                  "video": video_path.name, "status": "running"}
        try:
            obs, _ = env.reset(seed=args.seed + episode)
            policy.reset()
            with av.open(str(video_path), "w") as container:
                stream = container.add_stream("libx264", rate=args.control_freq)
                stream.width, stream.height, stream.pix_fmt = 512, 256, "yuv420p"

                def record(observation):
                    # 视频并排显示两个真实相机视角，不使用数据集视频冒充模型执行。
                    views = [np.ascontiguousarray(observation["pixels"][key][::-1, ::-1])
                             for key in ("image", "image2")]
                    frame = av.VideoFrame.from_ndarray(np.concatenate(views, axis=1), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

                record(obs)
                with torch.inference_mode(), open(output / f"actions_{episode:03d}.jsonl", "w") as log:
                    for step in range(args.max_steps):
                        batch = env_processor(preprocess_observation(batch_observation(obs)))
                        if args.image_rotation == 0:
                            for key in config.image_features:
                                batch[key] = torch.flip(batch[key], dims=[2, 3])
                        batch["task"] = [task_text]
                        batch = preprocess(batch)
                        normalized_action = policy.select_action(batch)
                        action = postprocess(normalized_action).squeeze(0).cpu().numpy()
                        if action.shape != (7,) or not np.isfinite(action).all():
                            raise ValueError(f"Invalid action: {action}")
                        executed = np.clip(action, env.action_space.low, env.action_space.high)
                        result["clipped_components"] += int(np.count_nonzero(executed != action))
                        obs, reward, terminated, truncated, info = env.step(executed)
                        record(obs)
                        result.update(steps=step + 1, success=bool(info.get("is_success", False)))
                        log.write(json.dumps({"step": step, "predicted": action.tolist(),
                                              "executed": executed.tolist(), "reward": float(reward),
                                              "success": result["success"]}) + "\n")
                        if step % 25 == 0:
                            print(f"episode={episode} step={step + 1} success={result['success']}", flush=True)
                        if terminated or truncated or result["success"]:
                            break
                for packet in stream.encode():
                    container.mux(packet)
            result["status"] = "completed"
        except Exception as exc:
            result.update(status="error", error=str(exc))
            raise
        finally:
            env.close()
            report["episodes"].append(result)
            (output / "results.json").write_text(json.dumps(report, indent=2))
        print("video:", video_path, "success:", result["success"], flush=True)


if __name__ == "__main__":
    main()
