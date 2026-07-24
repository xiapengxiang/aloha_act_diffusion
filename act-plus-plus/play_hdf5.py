import argparse
import os

import cv2
import h5py
import numpy as np

from constants import DT


PREFERRED_CAMERA_ORDER = ('cam_left_wrist', 'cam_right_wrist', 'cam_high', 'left_wrist', 'right_wrist', 'top')


def has_display():
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def resolve_dataset_path(args):
    if args['file_path'] is not None:
        return args['file_path']

    dataset_dir = args['dataset_dir']
    if dataset_dir is None:
        raise ValueError('Either --file_path or --dataset_dir must be provided')

    episode_prefix = 'mirror_episode' if args['ismirror'] else 'episode'
    episode_name = f'{episode_prefix}_{args["episode_idx"]}.hdf5'
    return os.path.join(dataset_dir, episode_name)


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
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        base_action = root['/base_action'][()] if '/base_action' in root else None
        camera_names = list(root['/observations/images'].keys())
        compress_len = root['/compress_len'][()] if compress else None

        image_dict = {}
        for cam_idx, cam_name in enumerate(camera_names):
            cam_compress_len = compress_len[cam_idx] if compress else None
            image_dict[cam_name] = decode_image_dataset(root[f'/observations/images/{cam_name}'], cam_compress_len)

        metadata = {
            'compress': compress,
            'sim': bool(root.attrs.get('sim', False)),
            'state_dim': int(root.attrs.get('state_dim', qpos.shape[1])),
            'action_dim': int(root.attrs.get('action_dim', action.shape[1])),
            'base_action_dim': int(root.attrs.get('base_action_dim', 0)),
            'camera_names': camera_names,
        }

    return qpos, qvel, action, base_action, image_dict, metadata


def get_camera_order(camera_names):
    ordered = [cam_name for cam_name in PREFERRED_CAMERA_ORDER if cam_name in camera_names]
    ordered.extend(sorted(cam_name for cam_name in camera_names if cam_name not in ordered))
    return ordered


def pad_to_height(image, target_height):
    height, width, channels = image.shape
    if height == target_height:
        return image
    pad_height = target_height - height
    padding = np.zeros((pad_height, width, channels), dtype=image.dtype)
    return np.concatenate([image, padding], axis=0)


def merge_frame(image_dict, frame_idx, camera_order):
    frames = [image_dict[cam_name][frame_idx] for cam_name in camera_order]
    max_height = max(frame.shape[0] for frame in frames)
    padded_frames = [pad_to_height(frame, max_height) for frame in frames]
    merged_rgb = np.concatenate(padded_frames, axis=1)
    merged_bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)
    return merged_bgr


def save_video(image_dict, camera_order, fps, output_path, max_frames=None):
    total_frames = min(len(image_dict[camera_order[0]]), max_frames) if max_frames is not None else len(image_dict[camera_order[0]])
    first_frame = merge_frame(image_dict, 0, camera_order)
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_idx in range(total_frames):
        writer.write(merge_frame(image_dict, frame_idx, camera_order))
    writer.release()
    print(f'Saved video to: {output_path}')


def display_video(image_dict, camera_order, fps, max_frames=None):
    total_frames = min(len(image_dict[camera_order[0]]), max_frames) if max_frames is not None else len(image_dict[camera_order[0]])
    delay_ms = max(1, int(round(1000 / fps)))
    for frame_idx in range(total_frames):
        frame = merge_frame(image_dict, frame_idx, camera_order)
        cv2.imshow('play_hdf5', frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord('q')):
            break
    cv2.destroyAllWindows()


def print_summary(dataset_path, qpos, qvel, action, base_action, image_dict, metadata):
    print(f'Dataset: {dataset_path}')
    print(f'compress={metadata["compress"]} sim={metadata["sim"]}')
    print(f'qpos shape: {qpos.shape}')
    print(f'qvel shape: {qvel.shape}')
    print(f'action shape: {action.shape}')
    if base_action is not None:
        print(f'base_action shape: {base_action.shape}')
    print(f'state_dim={metadata["state_dim"]} action_dim={metadata["action_dim"]} base_action_dim={metadata["base_action_dim"]}')
    for cam_name, frames in image_dict.items():
        print(f'{cam_name} shape: {frames.shape}')


def main(args):
    dataset_path = resolve_dataset_path(args)
    qpos, qvel, action, base_action, image_dict, metadata = load_hdf5_episode(dataset_path)
    print_summary(dataset_path, qpos, qvel, action, base_action, image_dict, metadata)

    camera_order = get_camera_order(metadata['camera_names'])
    fps = args['fps'] if args['fps'] is not None else int(round(1 / DT))
    can_display = has_display()
    output_path = args['output_path']
    if output_path is None:
        output_path = dataset_path.replace('.hdf5', '_playback.mp4')

    if args['save_video']:
        save_video(image_dict, camera_order, fps, output_path, args['max_frames'])

    if not args['no_display'] and not can_display:
        print('No display detected on this server. Skipping OpenCV playback.')
        print(f'Use --save_video to export a video file, for example: {output_path}')

    if not args['no_display'] and can_display:
        display_video(image_dict, camera_order, fps, args['max_frames'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play or export ACT-format HDF5 episodes.')
    parser.add_argument('--file_path', type=str, default=None, help='Path to a single .hdf5 episode file.')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Directory containing episode_*.hdf5 files.')
    parser.add_argument('--episode_idx', type=int, default=0, help='Episode index when using --dataset_dir.')
    parser.add_argument('--ismirror', action='store_true', help='Load mirror_episode_* when using --dataset_dir.')
    parser.add_argument('--fps', type=int, default=None, help='Playback/export FPS. Defaults to 1 / constants.DT.')
    parser.add_argument('--max_frames', type=int, default=None, help='Limit the number of frames to display or export.')
    parser.add_argument('--save_video', action='store_true', help='Save a merged MP4 playback video.')
    parser.add_argument('--output_path', type=str, default=None, help='Output path for saved video.')
    parser.add_argument('--no_display', action='store_true', help='Skip on-screen playback.')
    main(vars(parser.parse_args()))