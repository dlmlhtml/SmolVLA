import json

import draccus
import torch
from huggingface_hub import hf_hub_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


model_id = "lerobot/smolvla_base"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)

# 1. 加载模型
# Avoid draccus argparse's Python 3.14 union-type incompatibility.
with open(hf_hub_download(model_id, "config.json")) as f:
    config_data = json.load(f)
assert config_data.pop("type") == "smolvla"
config_data["device"] = str(device)
config = draccus.decode(SmolVLAConfig, config_data)
policy = SmolVLAPolicy.from_pretrained(model_id, config=config).to(device).eval()

# 2. 创建模型对应的输入/输出处理器   
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={
        "device_processor": {
            "device": str(device)
        }
    },
)

# 3. 加载一个官方机器人数据集
dataset = LeRobotDataset("lerobot/libero", episodes=[0], video_backend="pyav")

print("dataset loaded")
print("dataset length:", len(dataset))

print("\n=== DATASET INFO ===")
print("FPS:", dataset.meta.fps)
print("Features:")
for key, feature in dataset.meta.features.items():
    print(key, feature)

print("\n=== FIRST 3 FRAMES ===")
previous_timestamp = None
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    timestamp = sample["timestamp"].item()
    print(f"\n--- sample {i} ---")
    print("episode:", sample["episode_index"].item())
    print("frame:", sample["frame_index"].item())
    print("timestamp:", timestamp)
    if previous_timestamp is not None:
        print("time delta:", timestamp - previous_timestamp, "expected:", 1 / dataset.meta.fps)
    previous_timestamp = timestamp
    print("task:", sample["task"])
    print("state:", sample["observation.state"])
    print("expert action:", sample["action"])

# 4. 先拿第一帧
frame = dict(dataset[0])

print("\nframe keys:")
for key, value in frame.items():
    if hasattr(value, "shape"):
        print(key, value.shape)
    else:
        print(key, type(value))

# 5. preprocessing
# Map LIBERO's two camera views to the checkpoint's input names.
# This is an inference smoke test, not a LIBERO-adapted control policy:
# the checkpoint outputs 6 action dimensions while LIBERO uses 7.
frame["observation.images.camera1"] = frame.pop("observation.images.image")
frame["observation.images.camera2"] = frame.pop("observation.images.image2")
frame.pop("action", None)  # Ground-truth actions are not inference inputs.
batch = preprocess(frame)

print("\nINPUT TO VLA")
for key, value in batch.items():
    if hasattr(value, "shape"):
        print(key, value.shape, value.dtype)
    else:
        print(key, type(value))

# 6. Inspect the full model output chunk before postprocessing.
with torch.inference_mode():
    chunk = policy.predict_action_chunk(batch)

print("chunk shape:", chunk.shape)
print("first 3 actions:")
print(chunk[0, :3])
assert chunk.shape == (1, policy.config.chunk_size, policy.config.output_features["action"].shape[0])
assert torch.isfinite(chunk).all(), "Inference produced non-finite actions"
print("chunk inference passed (before postprocessing)")
