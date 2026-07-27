"""Standalone ACT inference helper and smoke test.

This script provides:
- ActInference: load an ACT checkpoint + config and run inference on observations
- a smoke test that loads a batch from training data and checks output shape / error
"""

import argparse
import os
import pickle
import sys
import time
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from constants import SIM_TASK_CONFIGS, TASK_CONFIGS
from detr.main import get_args_parser
from detr.models.latent_model import Latent_Model_Transformer
from policy import ACTPolicy, DiffusionPolicy


def decode_image_dataset(image_dataset, compress_len=None):
    if compress_len is None:
        return image_dataset[()]

    decoded_images = []
    raw_images = image_dataset[()]
    for frame_idx, encoded_image in enumerate(raw_images):
        image_len = int(compress_len[frame_idx])
        decoded = cv2.imdecode(encoded_image[:image_len], cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f'Failed to decode compressed image at frame {frame_idx}')
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        decoded_images.append(decoded)
    return np.stack(decoded_images, axis=0)


def load_hdf5_episode(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')

    with h5py.File(dataset_path, 'r') as root:
        compress = bool(root.attrs.get('compress', False))
        qpos = root['/observations/qpos'][()]
        action = root['/action'][()]
        camera_names = list(root['/observations/images'].keys())
        compress_len = root['/compress_len'][()] if compress else None

        image_dict = {}
        for cam_idx, cam_name in enumerate(camera_names):
            cam_compress_len = compress_len[cam_idx] if compress else None
            image_dict[cam_name] = decode_image_dataset(root[f'/observations/images/{cam_name}'], cam_compress_len)

        metadata = {
            'compress': compress,
            'sim': bool(root.attrs.get('sim', False)),
            'camera_names': camera_names,
            'episode_len': int(qpos.shape[0]),
        }

    return qpos, action, image_dict, metadata


def find_latest_step_checkpoint(ckpt_dir):
    candidates = []
    for path in Path(ckpt_dir).glob('policy_step_*_seed_*.ckpt'):
        match = re.search(r'policy_step_(\d+)_seed_\d+\.ckpt$', path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def normalize_heatmap(heatmap):
    heatmap = heatmap - np.min(heatmap)
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def split_attention_by_camera(attention_vector, camera_shapes):
    if not camera_shapes:
        return []

    heights = [shape[0] for shape in camera_shapes]
    widths = [shape[1] for shape in camera_shapes]
    if len(set(heights)) != 1:
        raise ValueError(f'Expected the same attention height for all cameras, got {camera_shapes}')

    height = heights[0]
    width_total = sum(widths)
    combined = attention_vector.reshape(height, width_total)

    camera_heatmaps = []
    offset = 0
    for width in widths:
        camera_heatmaps.append(combined[:, offset:offset + width])
        offset += width
    return camera_heatmaps


def save_attention_overlay(image_dict, attention_weights, camera_shapes, camera_names, output_path):
    attention_vector = attention_weights.detach().float().mean(dim=1).mean(dim=1)[0]
    attention_vector = attention_vector[2:]
    camera_heatmaps = split_attention_by_camera(attention_vector.cpu().numpy(), camera_shapes)
    if not camera_heatmaps:
        return

    fig, axes = plt.subplots(2, len(camera_names), figsize=(4 * len(camera_names), 8))
    if len(camera_names) == 1:
        axes = np.array(axes).reshape(2, 1)

    for cam_idx, cam_name in enumerate(camera_names):
        image = np.asarray(image_dict[cam_name])
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32) / 255.0

        heatmap = normalize_heatmap(camera_heatmaps[cam_idx])
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        heatmap_tensor = F.interpolate(heatmap_tensor, size=image.shape[:2], mode='bilinear', align_corners=False)
        heatmap = heatmap_tensor[0, 0].cpu().numpy()

        axes[0, cam_idx].imshow(np.clip(image, 0.0, 1.0))
        axes[0, cam_idx].set_title(f'{cam_name} image')
        axes[0, cam_idx].axis('off')

        axes[1, cam_idx].imshow(np.clip(image, 0.0, 1.0))
        axes[1, cam_idx].imshow(heatmap, cmap='jet', alpha=0.45)
        axes[1, cam_idx].set_title(f'{cam_name} attention')
        axes[1, cam_idx].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


class ActInference:
    def __init__(self, ckpt_dir, ckpt_name=None, device=None):
        self.ckpt_dir = Path(ckpt_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('ACTInference currently expects a CUDA device because the ACT model is evaluated on GPU in this repo.')

        self.config = self._load_pickle('config.pkl')
        self.stats = self._load_pickle('dataset_stats.pkl')
        self.policy_config = self.config['policy_config']
        self.camera_names = self.config['camera_names']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.use_vq = bool(self.policy_config.get('vq', False))

        if ckpt_name is None:
            preferred = ['policy_best.ckpt', 'policy_last.ckpt']
            for candidate in preferred:
                candidate_path = self.ckpt_dir / candidate
                if candidate_path.exists():
                    ckpt_name = candidate
                    break
            if ckpt_name is None:
                latest_checkpoint = find_latest_step_checkpoint(self.ckpt_dir)
                if latest_checkpoint is None:
                    raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
                ckpt_name = latest_checkpoint.name

        self.ckpt_path = self.ckpt_dir / ckpt_name
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {self.ckpt_path}')

        self.model_args = self._build_model_args()
        original_argv = sys.argv
        try:
            sys.argv = [
                'policy_inference.py',
                '--ckpt_dir', str(self.ckpt_dir),
                '--policy_class', 'ACT',
                '--task_name', self.config['task_name'],
                '--seed', str(self.config['seed']),
                '--num_steps', str(self.config['num_steps']),
            ]
            self.model = ACTPolicy(self.policy_config)
        finally:
            sys.argv = original_argv
        loading_status = self.model.deserialize(torch.load(self.ckpt_path, map_location='cpu'))
        print(f'Loaded policy from {self.ckpt_path}: {loading_status}')
        self.model.eval()

        self.latent_model = None
        if self.use_vq:
            vq_dim = self.policy_config['vq_dim']
            vq_class = self.policy_config['vq_class']
            self.latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_path = self.ckpt_dir / 'latent_model_last.ckpt'
            if not latent_model_path.is_file():
                raise FileNotFoundError(f'VQ is enabled but latent model checkpoint is missing: {latent_model_path}')
            self.latent_model.deserialize(torch.load(latent_model_path, map_location='cuda'))
            self.latent_model.eval()
            self.latent_model.cuda()
            print(f'Loaded latent model from {latent_model_path}')

        self.model.cuda()
        self._warmup_after_setup()

    @torch.inference_mode()
    def _warmup_after_setup(self, warmup_steps=3, warmup_height=480, warmup_width=640):
        try:
            qpos = np.zeros((self.state_dim,), dtype=np.float32)
            images = {
                cam_name: np.zeros((warmup_height, warmup_width, 3), dtype=np.uint8)
                for cam_name in self.camera_names
            }

            print(f'Running synthetic warmup for {warmup_steps} steps...')
            t0 = time.perf_counter()
            for _ in range(warmup_steps):
                _ = self._predict_normalized_tensor(qpos, images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f'Warmup done: {warmup_steps} steps in {elapsed_ms:.2f} ms')
        except Exception as exc:
            print(f'Warmup skipped due to error: {exc}')

    def _build_model_args(self):
        parser = get_args_parser()
        args = parser.parse_args([
            '--ckpt_dir', str(self.ckpt_dir),
            '--policy_class', 'ACT',
            '--task_name', self.config['task_name'],
            '--seed', str(self.config['seed']),
            '--num_steps', str(self.config['num_steps']),
        ])
        for key, value in self.policy_config.items():
            setattr(args, key, value)
        setattr(args, 'camera_names', self.camera_names)
        setattr(args, 'state_dim', self.state_dim)
        setattr(args, 'action_dim', self.action_dim)
        return args

    def _load_pickle(self, filename):
        path = self.ckpt_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def preprocess_qpos(self, qpos, already_normalized=False):
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.shape[0] != self.state_dim:
            raise ValueError(f'Expected qpos dim {self.state_dim}, got {qpos.shape[0]}')
        if already_normalized:
            return torch.from_numpy(qpos).float().unsqueeze(0).cuda()
        qpos = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        return torch.from_numpy(qpos).float().unsqueeze(0).cuda()

    def preprocess_images(self, images):
        if isinstance(images, dict):
            ordered = [images[name] for name in self.camera_names]
        else:
            ordered = list(images)

        image_tensors = []
        reference_shape = None
        for image in ordered:
            image = np.asarray(image)
            if image.ndim != 3:
                raise ValueError(f'Expected HWC image, got shape {image.shape}')
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            if reference_shape is None:
                reference_shape = image.shape
            elif image.shape != reference_shape:
                image = cv2.resize(image, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_AREA)
                if image.ndim == 2:
                    image = image[:, :, None]
            image = rearrange(image, 'h w c -> c h w')
            image_tensors.append(image)

        stacked = np.stack(image_tensors, axis=0)
        stacked = torch.from_numpy(stacked / 255.0).float().unsqueeze(0).cuda()
        return stacked

    def _build_chunk_target(self, action_sequence, start_ts, chunk_size, is_sim):
        if is_sim:
            chunk = action_sequence[start_ts:start_ts + chunk_size]
        else:
            chunk = action_sequence[max(0, start_ts - 1):max(0, start_ts - 1) + chunk_size]

        target = np.zeros((chunk_size, action_sequence.shape[1]), dtype=np.float32)
        target[:len(chunk)] = chunk
        return torch.from_numpy(target).float().unsqueeze(0).cuda()

    def _denormalize_action(self, action_tensor):
        action_mean = torch.from_numpy(self.stats['action_mean']).float().to(action_tensor.device)
        action_std = torch.from_numpy(self.stats['action_std']).float().to(action_tensor.device)
        return action_tensor * action_std + action_mean

    @torch.inference_mode()
    def _predict_normalized_tensor(self, qpos, images, already_normalized=False, vq_sample=None):
        qpos_tensor = self.preprocess_qpos(qpos, already_normalized=already_normalized)
        image_tensor = self.preprocess_images(images)

        if self.use_vq:
            if vq_sample is None:
                vq_sample = self.latent_model.generate(1, temperature=1, x=None)
            action = self.model(qpos_tensor, image_tensor, vq_sample=vq_sample)
        else:
            action = self.model(qpos_tensor, image_tensor)
        return action

    @torch.inference_mode()
    def predict(self, qpos, images, already_normalized=False, vq_sample=None):
        t0 = time.perf_counter()
        action = self._predict_normalized_tensor(
            qpos,
            images,
            already_normalized=already_normalized,
            vq_sample=vq_sample,
        )
        action = self._denormalize_action(action)
        action_np = action.squeeze(0).detach().cpu().numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f'predict latency: {elapsed_ms:.2f} ms')
        return action_np

    @torch.inference_mode()
    def predict_with_attention(self, qpos, images, already_normalized=False, vq_sample=None):
        if not isinstance(self.model, ACTPolicy):
            raise NotImplementedError('Attention heatmaps are only supported for ACT in this helper script.')

        t0 = time.perf_counter()
        qpos_tensor = self.preprocess_qpos(qpos, already_normalized=already_normalized)
        image_tensor = self.preprocess_images(images)

        if self.use_vq:
            if vq_sample is None:
                vq_sample = self.latent_model.generate(1, temperature=1, x=None)
            action, _, _, _, _, attn_weights, camera_shapes = self.model.forward_with_attention(
                qpos_tensor,
                image_tensor,
                vq_sample=vq_sample,
            )
        else:
            action, _, _, _, _, attn_weights, camera_shapes = self.model.forward_with_attention(
                qpos_tensor,
                image_tensor,
            )

        action = self._denormalize_action(action)
        action_np = action.squeeze(0).detach().cpu().numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f'predict-with-attention latency: {elapsed_ms:.2f} ms')
        return action_np, attn_weights.detach().cpu(), camera_shapes

    @torch.inference_mode()
    def smoke_test_from_training_data(self, dataset_dir=None, batch_size=1, chunk_size=None, episode_hdf5=None, save_heatmaps=False, heatmap_dir=None):
        task_name = self.config['task_name']
        if chunk_size is None:
            chunk_size = self.policy_config['num_queries']

        if save_heatmaps and not isinstance(self.model, ACTPolicy):
            raise NotImplementedError('Smoke-test heatmaps are only supported for ACT attention in this script.')

        if save_heatmaps:
            if heatmap_dir is None:
                heatmap_dir = str(self.ckpt_dir / 'heatmaps')
            os.makedirs(heatmap_dir, exist_ok=True)

        if episode_hdf5 is None:
            if dataset_dir is None:
                if task_name in TASK_CONFIGS:
                    dataset_dir = TASK_CONFIGS[task_name]['dataset_dir']
                elif task_name in SIM_TASK_CONFIGS:
                    dataset_dir = SIM_TASK_CONFIGS[task_name]['dataset_dir']
                else:
                    raise ValueError(f'Cannot resolve dataset_dir for task_name={task_name}')
            episode_candidates = sorted(Path(dataset_dir).glob('episode_*.hdf5'))
            if not episode_candidates:
                raise FileNotFoundError(f'No episode_*.hdf5 files found in {dataset_dir}')
            selected_idx = int(np.random.randint(len(episode_candidates)))
            episode_hdf5 = str(episode_candidates[selected_idx])
            print(f'Randomly selected episode for smoke test: {episode_hdf5}')

        qpos_seq, action_seq, image_dict, metadata = load_hdf5_episode(episode_hdf5)
        self.camera_names = metadata['camera_names']
        self.stats = self._load_pickle('dataset_stats.pkl')

        total_steps = qpos_seq.shape[0]
        all_pred = []
        all_target = []
        step_l1 = []
        step_l2 = []
        episode_stem = Path(episode_hdf5).stem if episode_hdf5 is not None else 'episode'

        for ts in range(total_steps):
            qpos = qpos_seq[ts]
            images = {cam_name: image_dict[cam_name][ts] for cam_name in self.camera_names}
            target = self._build_chunk_target(action_seq, ts, chunk_size, metadata['sim'])

            # Compare deployed robot-space actions from predict() against dataset action targets.
            if save_heatmaps:
                pred_np, attn_weights, camera_shapes = self.predict_with_attention(qpos, images)
                heatmap_path = os.path.join(heatmap_dir, episode_stem, f'step_{ts:04d}.png')
                save_attention_overlay(images, attn_weights, camera_shapes, self.camera_names, heatmap_path)
            else:
                pred_np = self.predict(qpos, images)
            pred = torch.from_numpy(pred_np).float().unsqueeze(0).cuda()
            pred = pred[:, :target.shape[1]]

            all_pred.append(pred)
            all_target.append(target)
            step_l1.append(F.l1_loss(pred, target).item())
            step_l2.append(F.mse_loss(pred, target).item())

        pred_tensor = torch.cat(all_pred, dim=0)
        target_tensor = torch.cat(all_target, dim=0)
        mae = F.l1_loss(pred_tensor, target_tensor).item()
        mse = F.mse_loss(pred_tensor, target_tensor).item()

        print('Episode smoke test summary')
        print(f'  episode: {episode_hdf5}')
        print(f'  steps: {total_steps}')
        print(f'  chunk_size: {chunk_size}')
        print(f'  pred tensor shape:   {tuple(pred_tensor.shape)}')
        print(f'  target tensor shape: {tuple(target_tensor.shape)}')
        print(f'  mean step mae: {float(np.mean(step_l1)):.6f}')
        print(f'  mean step mse: {float(np.mean(step_l2)):.6f}')
        print(f'  whole-episode mae: {mae:.6f}')
        print(f'  whole-episode mse: {mse:.6f}')
        return {
            'episode_hdf5': episode_hdf5,
            'steps': total_steps,
            'pred_shape': tuple(pred_tensor.shape),
            'target_shape': tuple(target_tensor.shape),
            'mae': mae,
            'mse': mse,
        }


class DiffusionInference(ActInference):
    def __init__(self, ckpt_dir, ckpt_name=None, device=None):
        self.ckpt_dir = Path(ckpt_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('DiffusionInference currently expects a CUDA device because the Diffusion model is evaluated on GPU in this repo.')

        self.config = self._load_pickle('config.pkl')
        self.stats = self._load_pickle('dataset_stats.pkl')
        self.policy_config = self.config['policy_config']
        self.camera_names = self.config['camera_names']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.use_vq = False
        self.latent_model = None

        if ckpt_name is None:
            preferred = ['policy_best.ckpt', 'policy_last.ckpt']
            for candidate in preferred:
                candidate_path = self.ckpt_dir / candidate
                if candidate_path.exists():
                    ckpt_name = candidate
                    break
            if ckpt_name is None:
                latest_checkpoint = find_latest_step_checkpoint(self.ckpt_dir)
                if latest_checkpoint is None:
                    raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
                ckpt_name = latest_checkpoint.name

        self.ckpt_path = self.ckpt_dir / ckpt_name
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {self.ckpt_path}')

        self.model = DiffusionPolicy(self.policy_config)
        loading_status = self.model.deserialize(torch.load(self.ckpt_path, map_location='cpu'))
        print(f'Loaded diffusion policy from {self.ckpt_path}: {loading_status}')
        self.model.eval()
        self.model.cuda()
        self._warmup_after_setup()

    def _predict_normalized_tensor(self, qpos, images, already_normalized=False, vq_sample=None):
        qpos_tensor = self.preprocess_qpos(qpos, already_normalized=already_normalized)
        image_tensor = self.preprocess_images(images)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        action = self.model(qpos_tensor, image_tensor)
        return action

    def _denormalize_action(self, action_tensor):
        action_min = torch.from_numpy(self.stats['action_min']).float().to(action_tensor.device)
        action_max = torch.from_numpy(self.stats['action_max']).float().to(action_tensor.device)
        return ((action_tensor + 1.0) / 2.0) * (action_max - action_min) + action_min


def main():
    parser = argparse.ArgumentParser(description='ACT inference and smoke test helper')
    parser.add_argument('--inference_policy_class', type=str, default='ACT', choices=['ACT', 'Diffusion'], help='Policy type to load for inference and smoke testing')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory containing config.pkl and dataset_stats.pkl')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Checkpoint filename to load (defaults to policy_best.ckpt or policy_last.ckpt)')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Override dataset directory for smoke test')
    parser.add_argument('--episode_hdf5', type=str, default=None, help='Path to a single HDF5 episode for a full-episode smoke test')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for smoke test')
    parser.add_argument('--chunk_size', type=int, default=None, help='Override chunk size for smoke test')
    parser.add_argument('--smoke_test', action='store_true', help='Run a training-data smoke test after loading the policy')
    parser.add_argument('--save_heatmaps', action='store_true', help='Save transformer attention overlays during the smoke test')
    parser.add_argument('--heatmap_dir', type=str, default=None, help='Directory to store smoke-test attention overlays')
    parser.add_argument('--qpos', type=str, default=None, help='Comma-separated raw qpos for one-step inference')
    parser.add_argument('--image_hdf5', type=str, default=None, help='Path to one HDF5 episode for single-step inference')
    parser.add_argument('--timestep', type=int, default=0, help='Timestep to read from --image_hdf5')
    args = parser.parse_args()

    if args.inference_policy_class == 'ACT':
        inference = ActInference(args.ckpt_dir, ckpt_name=args.ckpt_name)
    else:
        inference = DiffusionInference(args.ckpt_dir, ckpt_name=args.ckpt_name)

    if args.smoke_test:
        inference.smoke_test_from_training_data(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            episode_hdf5=args.episode_hdf5,
            save_heatmaps=args.save_heatmaps,
            heatmap_dir=args.heatmap_dir,
        )
        return

    if args.qpos is not None and args.image_hdf5 is not None:
        qpos = np.array([float(x) for x in args.qpos.split(',')], dtype=np.float32)
        with h5py.File(args.image_hdf5, 'r') as root:
            images = {cam_name: root[f'/observations/images/{cam_name}'][args.timestep] for cam_name in inference.camera_names}
        actions = inference.predict(qpos, images)
        print(f'Predicted action shape: {tuple(actions.shape)}')
        print('Predicted actions are de-normalized robot-space values.')
        print(actions)
        return

    print('Loaded ACTInference. Use --smoke_test or provide --qpos and --image_hdf5 for direct inference.')


if __name__ == '__main__':
    main()