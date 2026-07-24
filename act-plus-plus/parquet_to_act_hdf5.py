"""Parquet -> ACT HDF5 conversion utility."""

import argparse
import glob
import h5py
import cv2
import numpy as np
import os
import pandas as pd
from typing import Dict, List


LEFT_IMAGE_SLICE = slice(0, 240)
RIGHT_IMAGE_SLICE = slice(240, 480)
HEAD_IMAGE_SLICE = slice(480, 720)
CAMERA_NAMES = ('cam_left_wrist', 'cam_right_wrist', 'cam_high')


def find_parquet_files(input_dir: str, recursive: bool, pattern: str) -> List[str]:
    search_pattern = os.path.join(input_dir, "**", pattern) if recursive else os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern, recursive=recursive))
    return [path for path in files if path.lower().endswith(".parquet")]


def inspect_with_pyarrow(file_path: str) -> Dict:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(file_path)
    schema = parquet_file.schema_arrow
    column_names = schema.names

    summary = {
        "file": file_path,
        "num_rows": parquet_file.metadata.num_rows,
        "num_row_groups": parquet_file.num_row_groups,
        "columns": column_names,
    }

    return summary


def inspect_with_pandas(file_path: str) -> Dict:
    import pandas as pd

    df = pd.read_parquet(file_path)
    column_names = list(df.columns)

    summary = {
        "file": file_path,
        "num_rows": len(df),
        "num_row_groups": "unknown (pandas backend)",
        "columns": column_names,
    }

    return summary


def inspect_parquet_file(file_path: str) -> Dict:
    try:
        return inspect_with_pyarrow(file_path)
    except Exception as pyarrow_err:
        print(f"[warn] pyarrow read failed for {file_path}: {pyarrow_err}")
        try:
            return inspect_with_pandas(file_path)
        except Exception as pandas_err:
            raise RuntimeError(
                f"Failed to inspect parquet file {file_path}. "
                f"pyarrow error: {pyarrow_err}; pandas error: {pandas_err}"
            )


def collect_dataset_keys(parquet_files: List[str]) -> List[str]:
    all_keys = set()
    for file_path in parquet_files:
        summary = inspect_parquet_file(file_path)
        all_keys.update(summary['columns'])
    return sorted(all_keys)


def ensure_float_array(value, dtype=np.float32):
    array = np.asarray(value, dtype=dtype)
    return np.squeeze(array)


def ensure_vector_length(value, expected_dim, value_name):
    vector = ensure_float_array(value).reshape(-1).astype(np.float32)
    if vector.shape[0] != expected_dim:
        raise ValueError(f'Expected {value_name} dim {expected_dim}, got {vector.shape[0]}')
    return vector


def get_hand_scalar(value, hand_index):
    hand_value = ensure_float_array(value)
    if hand_value.ndim == 0:
        if hand_index != 0:
            raise ValueError('Expected per-hand xhand value, got scalar')
        return float(hand_value)
    if hand_value.shape[0] <= hand_index:
        raise ValueError(f'Hand index {hand_index} out of bounds for shape {hand_value.shape}')
    selected = np.asarray(hand_value[hand_index]).reshape(-1)
    if selected.size == 0:
        raise ValueError(f'Empty hand value for index {hand_index}')
    return float(selected[0])


def decode_stacked_image(image_value):
    if isinstance(image_value, (bytes, bytearray)):
        encoded = np.frombuffer(image_value, dtype=np.uint8)
    else:
        encoded = np.asarray(image_value, dtype=np.uint8).reshape(-1)

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Failed to decode compressed_image_image')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_height, img_width = image.shape[:2]
    if img_width == 424 and img_height >= 720:
        left = cv2.rotate(image[0:240, :, :], cv2.ROTATE_180)
        right = image[240:480, :, :]
        head = image[480:720, :, :]
    elif img_width == 1064 and img_height >= 480:
        left = cv2.rotate(image[0:240, 0:424, :], cv2.ROTATE_180)
        right = image[240:480, 0:424, :]
        head = image[0:480, 424:1064, :]
    else:
        raise ValueError(
            f'Unsupported compressed_image_image layout: height={img_height}, width={img_width}'
        )

    return {
        'cam_left_wrist': left,
        'cam_right_wrist': right,
        'cam_high': head,
    }


def row_to_state_action(row):
    left_state = np.concatenate([
        ensure_vector_length(row['left_ee_pose'], 9, 'left_ee_pose'),
        np.array([get_hand_scalar(row['xhand_state'], 0)], dtype=np.float32),
    ]).astype(np.float32)
    right_state = np.concatenate([
        ensure_vector_length(row['right_ee_pose'], 9, 'right_ee_pose'),
        np.array([get_hand_scalar(row['xhand_state'], 1)], dtype=np.float32),
    ]).astype(np.float32)

    hand_control = ensure_vector_length(row['xhand_control'], 2, 'xhand_control')
    left_action = ensure_vector_length(row['left_ee_target'], 9, 'left_ee_target')
    right_action = ensure_vector_length(row['right_ee_target'], 9, 'right_ee_target')

    qpos = np.concatenate([left_state, right_state], axis=0).astype(np.float32)
    action = np.concatenate([left_action, right_action, hand_control], axis=0).astype(np.float32)
    return qpos, action


def validate_dimensions(qpos, action):
    if qpos.shape[-1] != 20:
        raise ValueError(f'Expected qpos dim 20, got {qpos.shape[-1]}')
    if action.shape[-1] != 20:
        raise ValueError(f'Expected action dim 20, got {action.shape[-1]}')


def write_episode(output_path, qpos, qvel, action, image_dict, source_path):
    max_timesteps = qpos.shape[0]
    image_height, image_width, channels = image_dict[CAMERA_NAMES[0]].shape[1:]

    with h5py.File(output_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        root.attrs['base_action_dim'] = 0
        root.attrs['state_dim'] = qpos.shape[1]
        root.attrs['action_dim'] = action.shape[1]
        root.attrs['source_parquet'] = source_path

        obs = root.create_group('observations')
        images = obs.create_group('images')
        obs.create_dataset('qpos', data=qpos, dtype='float32')
        obs.create_dataset('qvel', data=qvel, dtype='float32')
        root.create_dataset('action', data=action, dtype='float32')
        for cam_name in CAMERA_NAMES:
            images.create_dataset(
                cam_name,
                data=image_dict[cam_name],
                dtype='uint8',
                chunks=(1, image_height, image_width, channels),
            )


def convert_parquet_file(file_path, output_path):
    dataframe = pd.read_parquet(file_path)
    required_columns = {
        'left_ee_pose',
        'right_ee_pose',
        'left_ee_target',
        'right_ee_target',
        'xhand_state',
        'xhand_control',
        'compressed_image_image',
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise KeyError(f'Missing required columns in {file_path}: {missing_columns}')

    qpos_list = []
    action_list = []
    image_buffer = {cam_name: [] for cam_name in CAMERA_NAMES}

    for row in dataframe.to_dict(orient='records'):
        qpos, action = row_to_state_action(row)
        validate_dimensions(qpos, action)
        images = decode_stacked_image(row['compressed_image_image'])

        qpos_list.append(qpos)
        action_list.append(action)
        for cam_name in CAMERA_NAMES:
            image_buffer[cam_name].append(images[cam_name])

    qpos_array = np.stack(qpos_list, axis=0).astype(np.float32)
    action_array = np.stack(action_list, axis=0).astype(np.float32)
    qvel_array = np.zeros_like(qpos_array, dtype=np.float32)
    image_dict = {
        cam_name: np.stack(image_buffer[cam_name], axis=0).astype(np.uint8)
        for cam_name in CAMERA_NAMES
    }

    write_episode(output_path, qpos_array, qvel_array, action_array, image_dict, file_path)
    return qpos_array.shape[0]


def convert_to_act_hdf5(parquet_files, output_dir, warn_len_below=2, skip_len_below=0):
    os.makedirs(output_dir, exist_ok=True)
    episode_lengths = []
    failed_files = []
    short_episodes = []
    skipped_short = []
    for episode_idx, file_path in enumerate(parquet_files):
        output_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
        print(f'Converting {file_path} -> {output_path}')
        try:
            episode_length = convert_parquet_file(file_path, output_path)
            if warn_len_below > 0 and episode_length < warn_len_below:
                short_episodes.append((output_path, episode_length, file_path))
            if skip_len_below > 0 and episode_length < skip_len_below:
                os.remove(output_path)
                skipped_short.append((output_path, episode_length, file_path))
                continue
            episode_lengths.append(episode_length)
        except Exception as exc:
            failed_files.append(file_path)
            print(f'[failed] {file_path}: {exc}')

    print('\nConversion summary')
    print(f'Successfully converted: {len(episode_lengths)} / {len(parquet_files)}')
    if episode_lengths:
        lengths = np.asarray(episode_lengths, dtype=np.int32)
        print(f'Average length: {float(lengths.mean()):.2f}')
        print(f'Min length: {int(lengths.min())}')
        print(f'Max length: {int(lengths.max())}')
    if failed_files:
        print('Failed files:')
        for file_path in failed_files:
            print(f'  - {file_path}')
    if short_episodes:
        print(f'Short episodes (< {warn_len_below} frames): {len(short_episodes)}')
        for output_path, length, source_file in short_episodes[:50]:
            print(f'  - {output_path}: len={length}, source={source_file}')
    if skipped_short:
        print(f'Skipped short episodes (< {skip_len_below} frames): {len(skipped_short)}')


def main():
    parser = argparse.ArgumentParser(description="Convert parquet robot data to ACT HDF5 format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing parquet files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Target folder for converted ACT HDF5 files.")
    parser.add_argument("--pattern", type=str, default="*.parquet", help="Filename pattern to match parquet files.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subfolders.")
    parser.add_argument("--warn_len_below", type=int, default=2, help="Report episodes shorter than this frame count.")
    parser.add_argument("--skip_len_below", type=int, default=0, help="Delete converted episodes shorter than this frame count (0 disables skipping).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    parquet_files = find_parquet_files(args.input_dir, recursive=args.recursive, pattern=args.pattern)
    if not parquet_files:
        print(f"No parquet files found in {args.input_dir} (pattern={args.pattern}, recursive={args.recursive}).")
        return

    dataset_keys = collect_dataset_keys(parquet_files)
    print(f"Found {len(parquet_files)} parquet files to convert.")
    print(f"Keys ({len(dataset_keys)}):")
    for key in dataset_keys:
        print(f"  - {key}")

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir + "_act_hdf5"
    convert_to_act_hdf5(parquet_files, output_dir, warn_len_below=args.warn_len_below, skip_len_below=args.skip_len_below)


if __name__ == "__main__":
    main()
