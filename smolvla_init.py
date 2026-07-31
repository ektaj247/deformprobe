"""
smolvla_init.py — minimal working SmolVLA inference on a single image.

Proves the VLA pipeline runs end to end: load model -> read an image ->
preprocess (tokenize + normalize) -> generate an action chunk.

NOTE ON OUTPUT: SmolVLA-base outputs SO-100/SO-101 arm JOINT commands
(6 values per step), not image coordinates. The action is not spatially
interpretable on arbitrary photos — this script's purpose is to confirm the
pipeline works, not to produce a grasp point. Mapping joint space -> pixels
would require the robot's kinematics and camera calibration (future work).

Environment (Colab, T4 GPU):
    pip install "av==12.3.0" "lerobot[smolvla]"
    # then RESTART the runtime before running (av binds only after restart)

Usage:
    python smolvla_init.py --image data/images/cloth_01.jpg \
                            --task "fold the towel in half"
"""

import argparse
import numpy as np
import torch
from PIL import Image

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors

MODEL_ID = "lerobot/smolvla_base"
# dataset used only to source normalization stats for the preprocessor
STATS_REPO = "lerobot/svla_so101_pickplace"
IMG_SIZE = 256


def load_image(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # (3,H,W)
    return t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", default="fold the towel in half")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).eval().to(device)
    print("model loaded on", device)

    img_t = load_image(args.image, device)
    print("image ready:", args.image, tuple(img_t.shape))

    meta = LeRobotDatasetMetadata(STATS_REPO)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, dataset_stats=meta.stats
    )

    # SmolVLA-base expects 3 camera views + a 6-dim joint state.
    # We have one photo (fed to all 3 views) and no robot (zero state).
    raw_obs = {
        "observation.images.camera1": img_t,
        "observation.images.camera2": img_t.clone(),
        "observation.images.camera3": img_t.clone(),
        "observation.state": torch.zeros(6),
        "task": args.task,
    }

    policy.reset()
    policy_input = preprocessor(raw_obs)
    with torch.inference_mode():
        action = policy.select_action(policy_input)
    action = postprocessor(action)

    print("\nACTION CHUNK SHAPE:", tuple(action.shape))
    print("ACTION (SO-100 joint deltas):\n", action)


if __name__ == "__main__":
    main()