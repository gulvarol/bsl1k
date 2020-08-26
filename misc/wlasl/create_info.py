"""Script to create the info file metadata for the WLASL dataset.

Example usage:
    ipython misc/wlasl/create_info.py
"""
import argparse
import json
import os
import pickle as pkl

import cv2

from beartype import beartype
from pathlib import Path
from download_wlasl import construct_output_path


def _get_video_info(rgb_path: Path):
    if os.path.exists(rgb_path):
        cap = cv2.VideoCapture(str(rgb_path))
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


@beartype
def main(dataset_path: Path, anno_file: str):
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

    set_dict = {"train": 0, "val": 1, "test": 2}

    json_file = os.path.join(dataset_path, "info", anno_file)
    with open(json_file) as ipf:
        content = json.load(ipf)

    labels = [ent["gloss"] for ent in content]

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

    cnt = 0
    for ent in content:
        gloss = ent["gloss"]
        for inst in ent["instances"]:
            output_filename = construct_output_path(
                output_dir=dataset_path / "videos_original",
                inst=inst,
            )
            if os.path.exists(output_filename):
                cnt += 1

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
                    sign_text = gloss
                    # assert(words_to_id[sign_text] == row['label'])
                    # if not (video_fps == row['fps']):
                    #     print(s, i, video_fps, row['fps'])
                    data["videos"]["videos_original"]["T"].append(video_res_t)
                    data["videos"]["videos_original"]["W"].append(video_res_w)
                    data["videos"]["videos_original"]["H"].append(video_res_h)
                    data["videos"]["videos_original"]["duration_sec"].append(
                        video_duration_sec
                    )
                    data["videos"]["videos_original"]["fps"].append(video_fps)

                    data["videos"]["word"].append(sign_text)
                    data["videos"]["word_id"].append(words_to_id[sign_text])

                    data["videos"]["split"].append(set_dict[inst["split"]])
                    name = os.path.join(
                        inst["split"], os.path.basename(output_filename)
                    )
                    data["videos"]["name"].append(name)

                    data["videos"]["box"].append(inst["bbox"])
                    data["videos"]["signer_id"].append(inst["signer_id"])

    pkl.dump(data, open(info_file, "wb"))


if __name__ == "__main__":
    description = "Save a pickle file for the msasl dataset to be used at training."
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--dataset_path",
        type=Path,
        default="data/wlasl/",
        help="Path to the dataset",
    )
    p.add_argument(
        "--anno_file",
        type=str,
        default="WLASL_v0.3.json",
        choices=["WLASL_v0.1.json", "WLASL_v0.3.json"],
        help="dataset file"
    )
    main(**vars(p.parse_args()))
