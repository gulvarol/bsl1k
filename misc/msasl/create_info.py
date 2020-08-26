"""Script to create the info file metadata for the MSASL dataset.

Example usage:
    ipython misc/msasl/create_info.py
"""
import argparse
import os
from pathlib import Path
import pickle as pkl

import cv2
import pandas as pd

from download_msasl import construct_video_filename, parse_annotations


def _get_video_info(rgb_path):
    if os.path.exists(rgb_path):
        cap = cv2.VideoCapture(rgb_path)
        video_res_t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

        if video_res_t != count:
            print(video_res_t, count)
            video_res_t = count

        video_res_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_res_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_res_t == 0 or video_res_w == 0 or video_res_h == 0 or video_fps == 0:
            return [None] * 5
        else:
            video_duration_sec = float(video_res_t) / video_fps
        return video_res_t, video_res_w, video_res_h, video_fps, video_duration_sec
    else:
        return [None] * 5


def main(dataset_path: Path, trim_format="%06d"):
    info_file = dataset_path / "info" / "info.pkl"
    dict_file = open(dataset_path / "info" / "words.txt", "w")

    data = {}
    words = set()
    data["videos"] = {}
    data["videos"]["name"] = []  # Our naming convention (unique ID for a video)
    data["videos"]["word"] = []
    data["videos"]["word_id"] = []
    data["videos"]["split"] = []  # 0: train, 1: val, 2: test

    # Resolution info
    data["videos"]["videos_original"] = {}
    data["videos"]["videos_original"]["T"] = []
    data["videos"]["videos_original"]["W"] = []
    data["videos"]["videos_original"]["H"] = []
    data["videos"]["videos_original"]["duration_sec"] = []
    data["videos"]["videos_original"]["fps"] = []

    # Extra annot
    data["videos"]["box"] = []
    data["videos"]["signer_id"] = []
    data["videos"]["signer"] = []

    sets = ["train", "val", "test"]
    set_dict = {"train": 0, "val": 1, "test": 2}

    labels_json = dataset_path / "info" / "MSASL_classes.json"
    labels = pd.read_json(labels_json)
    labels = labels[0]

    # Write to TXT file
    words_to_id = {}
    words = []
    for i, w in enumerate(labels):
        words_to_id[w] = i
        words.append(w)
        dict_file.write(f"{i:05d} {w}\n")
    dict_file.close()

    data["words"] = words
    data["words_to_id"] = words_to_id

    for s in sets:
        input_json = dataset_path / "info" / f"MSASL_{s}.json"
        output_dir = dataset_path / "videos_original" / s
        dataset = parse_annotations(input_json)
        print(f"{len(dataset)} items in the {s} set.")
        for i, row in dataset.iterrows():
            output_filename = construct_video_filename(output_dir, row, trim_format)
            assert os.path.exists(output_filename)

            # Video resolution information
            (
                video_res_t,
                video_res_w,
                video_res_h,
                video_fps,
                video_duration_sec,
            ) = _get_video_info(output_filename)
            # Indication that the video is readable
            if video_res_t:
                sign_text = row["text"]  # clean_text
                assert words_to_id[sign_text] == row["label"]
                # if not (video_fps == row['fps']):
                #     print(s, i, video_fps, row['fps'])
                if not (video_res_w == row["width"]):
                    print(s, i, video_res_w, row["width"])
                if not (video_res_h == row["height"]):
                    print(s, i, video_res_h, row["height"])
                data["videos"]["videos_original"]["T"].append(video_res_t)
                data["videos"]["videos_original"]["W"].append(video_res_w)
                data["videos"]["videos_original"]["H"].append(video_res_h)
                data["videos"]["videos_original"]["duration_sec"].append(
                    video_duration_sec
                )
                data["videos"]["videos_original"]["fps"].append(video_fps)

                data["videos"]["word"].append(sign_text)
                data["videos"]["word_id"].append(row["label"])

                data["videos"]["split"].append(set_dict[s])
                name = os.path.join(s, os.path.basename(output_filename))
                data["videos"]["name"].append(name)

                data["videos"]["box"].append(row["box"])
                data["videos"]["signer_id"].append(row["signer_id"])
                data["videos"]["signer"].append(row["signer"])

    pkl.dump(data, open(info_file, "wb"))


if __name__ == "__main__":
    description = "Save a pickle file for the msasl dataset to be used at training."
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--dataset_path",
        type=Path,
        default="data/msasl/",
        help="Path to the dataset",
    )
    main(**vars(p.parse_args()))
