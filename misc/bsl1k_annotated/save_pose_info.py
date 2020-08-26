"""Helper script to store pose information for BSL-1K annotated.
"""
import argparse
import os
import pickle as pkl
from datetime import date

import numpy as np
from tqdm import tqdm


def tight_box_joints2D(openpose_output, ht, wt, region="body", tightest=False):
    """
    Returns the bounding box given the openpose 2D joint detections
    (height/width of top-left, height/width of bottom-right)
    - can be used for the body with region='body'
    - can be used for the mouth with region='mouth'
    - can be used for the face with region='face'
    """
    if region == "body":
        indices = []
        meta_key = "body"
    elif region == "mouth":
        # Mouth indices within 70 facial landmarks according to:
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob
        # /master/doc/media/keypoints_face.png
        indices = range(48, 68)
        meta_key = "face"
    elif region == "face":
        indices = range(70)
        meta_key = "face"
    # Tighest bounding box covering the joint positions
    xmin = np.inf
    ymin = np.inf
    xmax = 0
    ymax = 0
    for t in range(len(openpose_output)):
        if len(openpose_output[t]) == 0:
            continue
        if indices == []:
            x_values = openpose_output[t][meta_key][:, 0]
            y_values = openpose_output[t][meta_key][:, 1]
        else:
            x_values = openpose_output[t][meta_key][indices, 0]
            y_values = openpose_output[t][meta_key][indices, 1]
        # Filter out the zeros
        nonzero_loc_x = np.where(x_values != 0)[0]
        nonzero_loc_y = np.where(y_values != 0)[0]
        nonzero_ix = list(set(nonzero_loc_x).intersection(set(nonzero_loc_y)))
        if len(nonzero_ix) != 0:
            x_values = x_values[nonzero_ix]
            y_values = y_values[nonzero_ix]
        else:
            # If all zeros
            if region == "body":
                return [0.0, 0.0, 1.0, 1.0]
            elif region == "mouth" or region == "face":
                return [0.0, 0.0, 0.0, 0.0]
        x0 = x_values.min()
        y0 = y_values.min()
        x1 = x_values.max()
        y1 = y_values.max()
        if x0 < xmin:
            xmin = x0
        if y0 < ymin:
            ymin = y0
        if x1 > xmax:
            xmax = x1
        if y1 > ymax:
            ymax = y1

    # If for some reason none of the frames have pose detection,
    # use the whole frame [0, 0, 1, 1]
    if xmin == np.inf:
        xmin = 0
    if ymin == np.inf:
        ymin = 0
    if xmax == 0:
        xmax = wt
    if ymax == 0:
        ymax = ht

    if not tightest:
        # Resolution of the box
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # Slightly larger area to cover the head/feet of the human
        # Make sure extended points fall in the wt x ht img borders.
        xmin = max(0, xmin - 0.25 * box_width)  # left
        ymin = max(0, ymin - 0.35 * box_height)  # top
        xmax = min(xmax + 0.25 * box_width, wt)  # right
        ymax = min(ymax + 0.25 * box_height, ht)  # bottom

    return [ymin / ht, xmin / wt, ymax / ht, xmax / wt]


def get_episode_names(names, return_set=False):
    if return_set:
        out = set()
    else:
        out = []
    for name in names:
        parent_folder, base_name = name.split("--")
        parent_folder = os.path.basename(
            parent_folder
        )  # train/abortion/countryfile -> countryfile
        name_parts = base_name.split("_")
        base_name = name_parts[
            0
        ]  # 5975971110462486539_002020_002022.mp4 -> 5975971110462486539
        episode = f"{parent_folder}--{base_name}"
        # out = os.path.join(parent_folder, base_name)
        if return_set:
            out.add(episode)
        else:
            out.append(episode)
    return out


def get_openpose_interval(
    episode_pose_data,
    mouthing_time_sec,
    num_frames,
    # pose_keys=['body', 'face', 'lhnd', 'rhnd'],
    num_frames_before=64,
    num_frames_after=0,
    fps=25.0,
):
    start_time_sec = mouthing_time_sec - (num_frames_before / fps)
    # end_time_sec = mouthing_time_sec + (num_frames_after / fps)
    # This is not super accurate, there might be 1-frame offset
    # To make sure the numframes are the same, we pass that as input
    start_time_frame = int(round(start_time_sec * fps))
    end_time_frame = start_time_frame + num_frames  # round(end_time_sec * fps)

    openpose_interval = []
    for i in range(start_time_frame, end_time_frame):
        # ix = np.where(episode_pose_data['frame_idx'] == i)[0]
        # if len(ix) != 0
        #     ix = ix[0]
        # else:
        #     import pdb; pdb.set_trace()
        # It should be index i if it exists, otherwise the closest frame
        ix = abs(episode_pose_data["frame_idx"] - i).argmin()
        openpose_interval.append(episode_pose_data["pose_dicts"][ix])
    # 'body' [18x3], 'face' [70x3], 'lhnd' [21x3], 'rhnd' [21x3], 'signer' [1]
    return openpose_interval


def process_partition(dataset_path, path_openpose_bsl1k, partition):
    newinfofile = os.path.join(
        dataset_path, "info", "partitions", f"pose_{partition:02d}.pkl"
    )
    if os.path.exists(newinfofile):
        print(f"Already saved {newinfofile}")
        return
    infofile = os.path.join(dataset_path, "info", "info-annotated-videos-fixed.pkl")
    data = pkl.load(open(infofile, "rb"))
    videolist = [
        os.path.join(dataset_path, "videos", s.strip()) for s in data["videos"]["name"]
    ]
    video_episodes = get_episode_names(videolist)

    pkl_file = f"{path_openpose_bsl1k}-{partition:02d}-50.pkl"
    print(f"Loading {pkl_file}")
    pose_data = pkl.load(open(pkl_file, "rb"))
    # episode_keys = sorted(list(get_episode_names(videolist, return_set=True)))
    episode_keys = list(pose_data.keys())
    partition_ix = [i for i, e in enumerate(video_episodes) if e in episode_keys]
    bbox_data = []
    mouth_data = []
    face_data = []
    openpose_data = []
    ix_data = []
    for ix in tqdm(partition_ix):
        ht = data["videos"]["videos"]["H"][ix]
        wt = data["videos"]["videos"]["W"][ix]
        num_frames = data["videos"]["videos"]["T"][ix]

        openpose_interval = get_openpose_interval(
            episode_pose_data=pose_data[video_episodes[ix]],
            mouthing_time_sec=data["videos"]["mouthing_time"][ix],
            num_frames=num_frames,
        )

        assert len(openpose_interval) == num_frames
        bbox = tight_box_joints2D(openpose_interval, ht, wt)
        mouth = tight_box_joints2D(
            openpose_interval, ht, wt, region="mouth", tightest=True
        )
        face = tight_box_joints2D(
            openpose_interval, ht, wt, region="face", tightest=True
        )
        bbox_data.append(bbox)
        mouth_data.append(mouth)
        face_data.append(face)
        openpose_data.append(openpose_interval)
        ix_data.append(ix)

    # Append bbox info to info.pkl (and symlink manually)
    partition_data = {}
    partition_data["bbox"] = bbox_data
    partition_data["mouth"] = mouth_data
    partition_data["face"] = face_data
    partition_data["pose"] = openpose_data
    partition_data["ix"] = ix_data
    pkl.dump(partition_data, open(newinfofile, "wb"))


def gather_partitions(dataset_path):
    infofile = os.path.join(dataset_path, "info", "info-annotated-videos-fixed.pkl")
    info_data = pkl.load(open(infofile, "rb"))
    N = len(info_data["videos"]["name"])
    data = {}
    data["bbox"] = [None] * N
    data["mouth"] = [None] * N
    data["face"] = [None] * N
    data["pose"] = [None] * N
    running_partition_cnt = 0
    for partition in tqdm(range(50)):
        pkl_file = os.path.join(
            dataset_path, "info", "partitions", f"pose_{partition:02d}.pkl"
        )
        partition_data = pkl.load(open(pkl_file, "rb"))
        for x, i in enumerate(partition_data["ix"]):
            if partition_data["pose"][x] is None:
                import pdb

                pdb.set_trace()
            data["bbox"][i] = partition_data["bbox"][x]
            data["mouth"][i] = partition_data["mouth"][x]
            data["face"][i] = partition_data["face"][x]
            data["pose"][i] = partition_data["pose"][x]
        running_partition_cnt += len(partition_data["pose"])
        current_set_cnt = np.array([i is not None for i in data["pose"]]).sum()
        assert running_partition_cnt == current_set_cnt
    nan_cnt = np.array([i is None for i in data["pose"]]).sum()
    print(f"There are {nan_cnt} None poses.")

    today_str = date.today().strftime("%y.%m.%d")
    pkl_file = os.path.join(dataset_path, "info", f"pose_{today_str}.pkl")
    mouth_pkl_file = os.path.join(
        dataset_path, "info", f"mouth_bbox_{today_str}.pkl"
    )
    face_pkl_file = os.path.join(
        dataset_path, "info", f"face_bbox_{today_str}.pkl"
    )
    pkl.dump(data, open(pkl_file, "wb"))
    pkl.dump(data["mouth"], open(mouth_pkl_file, "wb"))
    pkl.dump(data["face"], open(face_pkl_file, "wb"))


def main(dataset_path, path_openpose_bsl1k):
    for partition in range(50):
        process_partition(dataset_path, path_openpose_bsl1k, partition)
    gather_partitions(dataset_path)


if __name__ == "__main__":
    description = "Helper script to save openpose outputs"
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--dataset_path",
        type=str,
        default="data/bsl1k_annotated",
        help="Root folder of the dataset.",
    )
    p.add_argument(
        "--path_openpose_bsl1k",
        type=str,
        help="Directory where openpose outputs for bsl1k videos are.",
    )
    main(**vars(p.parse_args()))
