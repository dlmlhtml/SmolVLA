"""Run one real SmolVLA training update; this is not a task-success evaluation."""

import json

import draccus
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    model_id = "lerobot/smolvla_base"
    with open(hf_hub_download(model_id, "config.json")) as f:
        config_data = json.load(f)
    assert config_data.pop("type") == "smolvla"
    config_data["device"] = device
    config = draccus.decode(SmolVLAConfig, config_data)

    dataset = LeRobotDataset("lerobot/libero", episodes=[0], video_backend="pyav")
    features = dataset_to_policy_features(dataset.meta.features)
    config.input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    config.output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    # The pretrained projections use padded max dimensions, so the full 7D action fits.
    assert config.output_features["action"].shape[0] <= config.max_action_dim
    dataset = LeRobotDataset(
        "lerobot/libero", episodes=[0], video_backend="pyav",
        delta_timestamps={"action": [i / dataset.meta.fps for i in range(config.chunk_size)]},
    )
    preprocess, _ = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    batch = preprocess(next(iter(DataLoader(dataset, batch_size=1, num_workers=0))))
    print("training action shape:", batch["action"].shape, flush=True)
    assert batch["action"].shape == (1, config.chunk_size, 7)

    policy = SmolVLAPolicy.from_pretrained(model_id, config=config, strict=True).to(device)
    policy.train()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    expert = policy.model.vlm_with_expert.lm_expert
    probe = next(p for p in expert.parameters() if p.requires_grad)
    before = probe.detach().clone()

    # The four operations that constitute one training update.
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = policy.forward(batch)
    assert torch.isfinite(loss), "Non-finite loss"
    print("loss:", loss.item(), flush=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0, error_if_nonfinite=True)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in expert.parameters())
    vlm = policy.model.vlm_with_expert.vlm
    assert all(not p.requires_grad and p.grad is None for p in vlm.parameters())
    optimizer.step()

    max_change = (probe.detach() - before).abs().max().item()
    assert max_change > 0, "Action expert parameter did not update"
    print("gradient norm:", grad_norm.item())
    print("expert parameter max change:", max_change)
    print("frozen VLM: no gradients")
    print("one training step passed; no checkpoint saved")


if __name__ == "__main__":
    main()
