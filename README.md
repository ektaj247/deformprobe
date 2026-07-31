# DeformProbe

**Does a vision-language-action model handle cloth the way it reasons about cloth?**

**How well do vision-language models reason about where to grasp cloth?**

DeformProbe evaluates how well models choose grasp points on deformable objects.
It focuses on cloth — the object class where such models are most likely to
struggle — and builds on the multiple-choice reasoning evaluation of ManipBench
(Seita et al., CoRL 2025), applied to a set of cloth images spanning a range of
states from flat to crumpled.

## Motivation

Vision-language models carry rich common-sense knowledge from internet
pretraining, but how well that knowledge transfers to precise, low-level
manipulation of deformable objects — with their complex, self-occluding
configurations — is not well characterized. DeformProbe measures grasp-point
reasoning across cloth states, as a step toward the longer-term question of
whether a policy's stated reasoning matches the actions it would actually
generate.

## Approach

Each cloth image is turned into a ManipBench-style multiple-choice question —
candidate grasp points labeled on the image — and a vision-language model is
asked which point to grasp for a given fold/flatten instruction. Answers are
scored against human-labeled grasp corners, broken down by cloth state
(flat → folded → crumpled → draped).

## Status

Reasoning pipeline working end to end (image → model → scored result), with
per-cloth-state accuracy. Building a harder, more discriminative question set
next.

## Pilot results (preliminary)

A first pass over 8 gradeable questions (2 excluded as too occluded to pose a
choice) using a Gemini flash model scored highly across all cloth states — a
coarse corner-vs-center format that does not yet discriminate, motivating harder
questions. Directional only (N=8).

## Repository

- `data/images/` — top-down cloth photos across 10 states
- `data/labels.json` — human-labeled corners, grasp points, and states
- `make_questions.py` — builds annotated MCQ images + `questions.json`
- `run_vlm.py` — runs a VLM over the questions and scores by cloth state
- `smolvla_hello.py` — minimal script that runs a small VLA (SmolVLA) on an image

## Reproduce

No GPU, free Gemini API tier:
`python make_questions.py` then `python run_vlm.py --model <gemini-flash-model>`.

<[Colab Notebook](https://colab.research.google.com/drive/1nSAhHmEdgWCeoEIvRq1gLj4WxJ16PzmO?usp=sharing)>

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

### Advising context (Seita / SLURM Lab)

- **Daniel Seita** — [homepage](https://danielseita.github.io/)
- **SLURM Lab @ USC** — [site](https://slurm-lab-usc.github.io/) · [github org](https://github.com/slurm-lab-usc)
- **GPT-Fabric** (foundation models for fabric manipulation) — [arXiv](https://arxiv.org/abs/2406.09640) · [folding repo](https://github.com/slurm-lab-usc/GPT-fabric-folding) · [smoothing repo](https://github.com/slurm-lab-usc/GPT-Fabric-Smoothing)

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