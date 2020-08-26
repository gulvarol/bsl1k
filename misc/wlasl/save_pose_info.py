import os
import json
import pickle as pkl
import argparse
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
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master
        # /doc/media/keypoints_face.png
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


def get_openpose_video(openpose_dir, num_frames):
    if not os.path.exists(openpose_dir):
        # print(f"Does not exist {openpose_dir}")
        return None
    # print(f"Loading {openpose_dir}")
    json_files = sorted(os.listdir(openpose_dir))
    assert num_frames == len(json_files)
    dim_info = {"body": 18, "lhnd": 21, "rhnd": 21, "face": 70}
    meta_keys = {
        "pose_keypoints_2d": "body",
        "hand_left_keypoints_2d": "lhnd",
        "hand_right_keypoints_2d": "rhnd",
        "face_keypoints_2d": "face",
    }
    out = []
    for json_file in json_files:
        frame_data = {}
        try:
            f = open(os.path.join(openpose_dir, json_file))
            content = json.load(f)
            # If there is no pose, fill with zeros
            if len(content["people"]) == 0:
                for k, v in meta_keys.items():
                    frame_data[v] = np.zeros((dim_info[v], 3))
            # Assume only 1 person
            else:
                pose_data = content["people"][0]
                for k, v in meta_keys.items():
                    frame_data[v] = np.array(pose_data[k]).reshape(dim_info[v], 3)
        except:
            # If cannot load json, fill with zeros
            for k, v in meta_keys.items():
                frame_data[v] = np.zeros((dim_info[v], 3))
        out.append(frame_data)
    assert len(out) == num_frames
    return out


def main(dataset_path):
    infofile = os.path.join(dataset_path, "info", "info.pkl")
    data = pkl.load(open(infofile, "rb"))
    videolist = [
        os.path.join(dataset_path, "videos-resized-256fps-256x256", s.strip())
        for s in data["videos"]["name"]
    ]
    N = len(videolist)

    splits = {0: "train", 1: "val", 2: "test"}

    pose_data = {}
    pose_data["bbox"] = []
    pose_data["mouth"] = []
    pose_data["face"] = []
    pose_data["pose"] = []
    ix_data = []
    for ix in tqdm(range(N)):
        ht = 256  # data['videos']['videos_360h_25fps']['H'][ix]
        wt = 256  # data['videos']['videos_360h_25fps']['W'][ix]
        num_frames = data["videos"]["videos_360h_25fps"]["T"][ix]
        split = splits[data["videos"]["split"][ix]]
        videoname = os.path.basename(videolist[ix])[:-4]
        openpose_dir = os.path.join(dataset_path, "wsl", split, videoname)
        openpose_data = get_openpose_video(openpose_dir, num_frames)

        if openpose_data is None:
            bbox = None
            mouth = None
            face = None
        else:
            bbox = tight_box_joints2D(openpose_data, ht, wt)
            mouth = tight_box_joints2D(
                openpose_data, ht, wt, region="mouth", tightest=True
            )
            face = tight_box_joints2D(
                openpose_data, ht, wt, region="face", tightest=True
            )
        pose_data["bbox"].append(bbox)
        pose_data["mouth"].append(mouth)
        pose_data["face"].append(face)
        pose_data["pose"].append(openpose_data)
        ix_data.append(ix)

    nan_cnt = np.array([i is None for i in pose_data["pose"]]).sum()
    print(f"There are {nan_cnt} None poses.")
    today_str = date.today().strftime("%y.%m.%d")
    pkl_file = os.path.join(dataset_path, "info", f"pose_{today_str}.pkl")
    mouth_pkl_file = os.path.join(dataset_path, "info", f"mouth_bbox_{today_str}.pkl")
    face_pkl_file = os.path.join(dataset_path, "info", f"face_bbox_{today_str}.pkl")
    pkl.dump(pose_data, open(pkl_file, "wb"))
    pkl.dump(pose_data["mouth"], open(mouth_pkl_file, "wb"))
    pkl.dump(pose_data["face"], open(face_pkl_file, "wb"))


if __name__ == "__main__":
    description = "Helper script to save openpose outputs"
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--dataset_path",
        type=str,
        default="/users/gul/datasets/wlasl",
        help="Root folder of the dataset.",
    )
    main(**vars(p.parse_args()))
