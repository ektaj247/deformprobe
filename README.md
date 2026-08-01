# DeformProbe

How well do vision-language models reason about where to grasp cloth?

DeformProbe evaluates how models pick grasp points on deformable objects. It
focuses on cloth, the object class where they're most likely to struggle, and
builds on the multiple-choice reasoning evaluation from ManipBench (Seita et
al., CoRL 2025), applied to cloth images ranging from flat to crumpled.

## Motivation

Vision-language models pick up a lot of common-sense knowledge from internet
pretraining. How much of that carries over to precise, low-level manipulation of
deformable objects is much less clear, since cloth self-occludes and its
configuration space is huge. DeformProbe measures grasp-point reasoning across
cloth states. The longer-term question behind it is whether a policy's stated
reasoning matches the actions it actually generates.

## Approach

Each cloth image becomes a ManipBench-style multiple-choice question, with
candidate grasp points labeled on the image, and a VLM is asked which point to
grasp for a given fold or flatten instruction. Answers are scored against
human-labeled grasp corners and broken down by cloth state (flat, folded,
crumpled, draped).

## Status

The reasoning pipeline runs end to end, from image to model to scored result,
with per-cloth-state accuracy. Next up is a harder, more discriminative question
set.

## Pilot results (preliminary)

A first pass over 8 gradeable questions (2 more were excluded as too occluded to
pose a real choice) with a Gemini flash model scored high on every cloth state.
The corner-vs-center format is too coarse to separate anything, which is why
harder questions come next. Directional only, N=8.

## Repository

- `data/images/`: top-down cloth photos across 10 states
- `data/labels.json`: human-labeled corners, grasp points, and states
- `make_questions.py`: builds annotated MCQ images plus `questions.json`
- `run_vlm.py`: runs a VLM over the questions and scores by cloth state
- `smolvla_hello.py`: minimal script that runs a small VLA (SmolVLA) on an image

## Reproduce

No GPU needed, works on the free Gemini API tier:

```
python make_questions.py
python run_vlm.py --model <gemini-flash-model>
```

There's also a [Colab notebook](https://colab.research.google.com/drive/1nSAhHmEdgWCeoEIvRq1gLj4WxJ16PzmO?usp=sharing).

## References and resources

### Core papers

- ManipBench (Seita et al., CoRL 2025): [project](https://manipbench.github.io/), [arXiv](https://arxiv.org/abs/2505.09698), [PMLR](https://proceedings.mlr.press/v305/zhao25a.html)
- π0.5: A VLA with Open-World Generalization, Physical Intelligence: [blog](https://www.pi.website/blog/pi05), [arXiv](https://arxiv.org/abs/2504.16054)
- π0: A VLA Flow Model for General Robot Control, Physical Intelligence: [blog](https://www.pi.website/blog/pi0), [arXiv](https://arxiv.org/abs/2410.24164)
- SmolVLA: An Efficient VLA for Affordable Robotics, Hugging Face: [arXiv](https://arxiv.org/abs/2506.01844), [model card](https://huggingface.co/lerobot/smolvla_base)

### Models and code

- [openpi](https://github.com/Physical-Intelligence/openpi): π0 / π0.5 / π0-FAST weights and code
- [LeRobot](https://github.com/huggingface/lerobot): hosts SmolVLA
- [FAST](https://arxiv.org/abs/2501.09747) action tokenizer
- [PaliGemma](https://arxiv.org/abs/2407.07726), π0's VLM backbone
- [OpenVLA](https://arxiv.org/abs/2406.09246), open 7B VLA, useful comparison point
- π0 / π0-FAST explainer on the [Hugging Face blog](https://huggingface.co/blog/pi0)

### Related work on fabric manipulation

- GPT-Fabric (foundation models for fabric manipulation): [arXiv](https://arxiv.org/abs/2406.09640), [folding repo](https://github.com/slurm-lab-usc/GPT-fabric-folding), [smoothing repo](https://github.com/slurm-lab-usc/GPT-Fabric-Smoothing)
- [SLURM Lab @ USC](https://slurm-lab-usc.github.io/) (Seita et al.)

### Simulation and perception tooling

- [SoftGym](https://github.com/Xingyu-Lin/softgym) for deformable sim, plus Seita's [install walkthrough](https://danieltakeshi.github.io/2021/02/20/softgym/)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) and [SAM 2](https://github.com/facebookresearch/sam2)
- [LIBERO](https://arxiv.org/abs/2306.03310), sim benchmark with a π0.5 checkpoint

### Compute and APIs (free tiers)

- [Google Colab](https://colab.research.google.com)
- [Google AI Studio](https://aistudio.google.com) for the free Gemini API used in the VLM baseline

## License

MIT
