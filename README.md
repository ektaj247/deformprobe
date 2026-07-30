# DeformProbe

**Does a vision-language-action model handle cloth the way it reasons about cloth?**

DeformProbe is an evaluation framework for studying whether a generalist robot
policy's stated reasoning about deformable objects matches the actions it
actually generates. It focuses on cloth — the object class where generalist
policies are most likely to break — and extends the multiple-choice reasoning
evaluation of ManipBench (Seita et al., CoRL 2025) toward direct action
generation.

## Motivation

Vision-language-action (VLA) models inherit rich common-sense knowledge from
internet-pretrained vision-language backbones. Whether that knowledge actually
reaches the action head — particularly for deformable objects, with their
complex, self-occluding configurations — is not well characterized. DeformProbe
measures the gap between what a policy *says* about a cloth manipulation task and
what it *does*.

## Approach

For each cloth scene, DeformProbe compares three signals:

- a language-level answer to a ManipBench-style question about where to grasp,
- the grasp implied by the action chunk the policy generates,
- a human-labeled set of acceptable grasp points.

The framework is developed against [SmolVLA](https://huggingface.co/lerobot/smolvla_base),
a 450M-parameter VLA that runs on consumer hardware and shares π0.5's
flow-matching, action-chunk architecture. π0.5 (Physical Intelligence) is the
target for a larger-scale extension.

## Status

Early development. Probe set and evaluation harness in progress.

## References and resources

### Core papers

- **ManipBench** — Seita et al., CoRL 2025 · [project](https://manipbench.github.io/) · [arXiv](https://arxiv.org/abs/2505.09698) · [PMLR](https://proceedings.mlr.press/v305/zhao25a.html)
- **π0.5: A VLA with Open-World Generalization** — Physical Intelligence · [blog](https://www.pi.website/blog/pi05) · [arXiv](https://arxiv.org/abs/2504.16054)
- **π0: A VLA Flow Model for General Robot Control** — Physical Intelligence · [blog](https://www.pi.website/blog/pi0) · [arXiv](https://arxiv.org/abs/2410.24164)
- **SmolVLA: An Efficient VLA for Affordable Robotics** — Hugging Face · [arXiv](https://arxiv.org/abs/2506.01844) · [model card](https://huggingface.co/lerobot/smolvla_base)

### Models and code

- **openpi** (π0 / π0.5 / π0-FAST weights + code) — [github](https://github.com/Physical-Intelligence/openpi)
- **LeRobot** (hosts SmolVLA) — [github](https://github.com/huggingface/lerobot)
- **FAST** action tokenizer — [arXiv](https://arxiv.org/abs/2501.09747)
- **PaliGemma** (π0's VLM backbone) — [arXiv](https://arxiv.org/abs/2407.07726)
- **OpenVLA** (open 7B VLA, useful comparison point) — [arXiv](https://arxiv.org/abs/2406.09246)
- π0 / π0-FAST explainer — [Hugging Face blog](https://huggingface.co/blog/pi0)

### Simulation and perception tooling

- **SoftGym** (deformable sim) — [github](https://github.com/Xingyu-Lin/softgym) · Seita's install walkthrough — [blog](https://danieltakeshi.github.io/2021/02/20/softgym/)
- **Grounded-Segment-Anything** — [github](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- **Grounding DINO** — [github](https://github.com/IDEA-Research/GroundingDINO)
- **Segment Anything (SAM)** — [github](https://github.com/facebookresearch/segment-anything) · **SAM 2** — [github](https://github.com/facebookresearch/sam2)
- **LIBERO** (sim benchmark; has a π0.5 checkpoint) — [arXiv](https://arxiv.org/abs/2306.03310)

### Compute and APIs (free tiers)

- **Google Colab** — [colab.research.google.com](https://colab.research.google.com)
- **Google AI Studio** (free Gemini API for the VLM baseline) — [aistudio.google.com](https://aistudio.google.com)

## License

MIT
