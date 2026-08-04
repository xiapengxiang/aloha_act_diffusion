# ALOHA ACT + Diffusion (Combined Repo)

This repository combines:
- act-plus-plus
- robomimic

It is intended for custom robot policy training and deployment with both ACT and Diffusion policies.

## 1) Environment Setup

Create and activate the conda environment:

```bash
conda create -n aloha python=3.8.10 -y
conda activate aloha
```

Install base dependencies following the act-plus-plus setup:

```bash
pip install torch torchvision
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops packaging h5py ipython
```

Install the local DETR package used by ACT:

```bash
cd act-plus-plus/detr
pip install -e .
cd ../..
```

Install local robomimic from this repository:

```bash
cd robomimic
pip install -e .
cd ..
```

Install extra data dependencies and pin NumPy:

```bash
pip install numpy==1.24.4 pandas pyarrow
```

Quick check:

```bash
python -c "import torch, numpy, pandas, pyarrow; print(torch.__version__, numpy.__version__)"
```

## 2) Data Layout

The training script reads dataset paths from task configs in:
- act-plus-plus/constants.py

Make sure your task config points to your HDF5 dataset directory and correct camera names.

## 3) Training

Go to training directory:

```bash
cd act-plus-plus
```

### Train ACT

--no_rollout_eval for avoiding rollout eval. Rollout evaluation is only supported for Aloha robot. Disable it with own robot.
--augment_images to simply randomize image input for better generalization.

```bash
python imitate_episodes.py \
  --task_name place_solar_panel \
  --ckpt_dir /fileStore/xpx_data/act_output/place_solar_panel \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_steps 20000 \
  --lr 1e-5 \
  --seed 0 \
  --no_rollout_eval \
  --augment_images
```

### Train Diffusion

```bash
python imitate_episodes.py \
  --task_name place_solar_panel \
  --ckpt_dir /fileStore/xpx_data/diffusion_output/place_solar_panel \
  --policy_class Diffusion \
  --chunk_size 20 \
  --batch_size 8 \
  --num_steps 20000 \
  --lr 1e-5 \
  --seed 0 \
  --no_rollout_eval
```

Notes:
- On headless servers, the GLFW DISPLAY warning is expected and usually harmless.
- If running on non-ALOHA real robot stack, keep --no_rollout_eval.

## 4) Inference and Smoke Test

This repo includes a unified inference script:
- act-plus-plus/policy_inference.py

It supports:
- ACT inference
- Diffusion inference
- full-episode smoke test
- synthetic warmup
- random episode selection for smoke test

### ACT smoke test

--save_heatmaps and --heatmap_dir to save attention heatmap.

```bash
cd act-plus-plus
python policy_inference.py \
  --inference_policy_class ACT \
  --ckpt_dir /fileStore/xpx_data/act_output/place_solar_panel \
  --ckpt_name policy_step_7000_kl_0.005.ckpt \
  --smoke_test \
  --save_heatmaps \
  --heatmap_dir /fileStore/xpx_data/aloha_data/ACT_heatmap
```

### Diffusion smoke test

```bash
cd act-plus-plus
python policy_inference.py \
  --inference_policy_class Diffusion \
  --ckpt_dir /fileStore/xpx_data/diffusion_output/place_solar_panel \
  --ckpt_name policy_step_12500_seed_0.ckpt \
  --smoke_test
```

### Single-step/chunk inference

```bash
cd act-plus-plus
python policy_inference.py \
  --inference_policy_class ACT \
  --ckpt_dir /fileStore/xpx_data/act_output/place_solar_panel \
  --ckpt_name policy_step_7000_kl_0.005.ckpt \
  --qpos "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0" \
  --image_hdf5 /fileStore/xpx_data/aloha_data/place_solar_panel/episode_0.hdf5 \
  --timestep 0
```

Output actions are de-normalized robot-space actions.
