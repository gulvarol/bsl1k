"""Script to create metadata for the manually verified portion of BSL-1K

ipython misc/bsl1k_annotated/create_info.py
"""
import os
import time
import pickle as pkl
import argparse
from pathlib import Path

import cv2
from zsvision.zs_beartype import beartype

from misc.bsl1k.extract_clips import (
    take_interval_from_peak,
    construct_video_filename
)


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
def main(
    data_dir: Path,
    anno_pkl_path: Path,
    canonical_1064_words: Path,
    prob_thres: float,
    video_folder: str,
    trim_format: str = "%06d",
):
    info_file = data_dir / "info" / f"info-{video_folder}.pkl"
    dict_file = open(data_dir / "info" / "words.txt", "w")

    data = {}
    words = set()
    data["videos"] = {}
    data["videos"]["name"] = []  # Our naming convention (unique ID for a video)
    data["videos"]["word"] = []
    data["videos"]["word_id"] = []
    data["videos"]["split"] = []  # 0: train, 1: val, 2: test

    # Resolution info
    data["videos"]["videos"] = {}
    data["videos"]["videos"]["T"] = []
    data["videos"]["videos"]["W"] = []
    data["videos"]["videos"]["H"] = []
    data["videos"]["videos"]["duration_sec"] = []
    data["videos"]["videos"]["fps"] = []

    # Extra annot
    data["videos"]["mouthing_time"] = []
    data["videos"]["mouthing_prob"] = []

    sets = ["test"]  # 'val',
    set_dict = {"train": 0, "val": 1, "test": 2}

    all_data = pkl.load(open(anno_pkl_path, "rb"))

    with open(canonical_1064_words, "rb") as f:
        words = set(pkl.load(f)["words"])

    # Only use train words from reference
    print(f"{len(words)} words")
    assert len(words) == 1064, "Expected 1064 words in vocab"
    mapping = {
        "airplane": "aeroplane",
        "center": "centre",
        "favor": "favour",
        "gray": "grey",
        "practice": "practise",
        "recognize": "recognise",
        "yogurt": "yoghurt",
    }
    # fix spellings to English
    updated_words = [mapping.get(word, word) for word in words]
    words = list(sorted(set(updated_words)))
    assert len(words) == 1064, "Expected 1064 words after fixing spellings"

    # Write to TXT file
    words_to_id = {}
    for i, w in enumerate(words):
        words_to_id[w] = i
        dict_file.write(f"{i:05d} {w}\n")
    dict_file.close()

    data["words"] = words
    data["words_to_id"] = words_to_id

    cnt = 0
    t0 = time.time()

    for s in sets:  # all_data.keys():
        for word_cnt, word in enumerate(all_data[s].keys()):
            if word in words_to_id:
                print(f"{time.time() - t0:0.2f} sec {s} {word_cnt} {word}")
                N = len(all_data[s][word]["names"])
                for i in range(N):
                    if all_data[s][word]["probs"][i] >= prob_thres:
                        start_time, end_time = take_interval_from_peak(
                            all_data[s][word]["times"][i]
                        )
                        output_filename = construct_video_filename(
                            output_dir=os.path.join(data_dir, video_folder),
                            set_name=s,
                            word=word,
                            name=all_data[s][word]["names"][i],
                            start_time=start_time,
                            end_time=end_time,
                            trim_format=trim_format,
                        )
                        if os.path.exists(output_filename):
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
                                data["videos"]["videos"]["T"].append(video_res_t)
                                data["videos"]["videos"]["W"].append(video_res_w)  # 480
                                data["videos"]["videos"]["H"].append(video_res_h)  # 480
                                data["videos"]["videos"]["duration_sec"].append(
                                    video_duration_sec
                                )
                                data["videos"]["videos"]["fps"].append(video_fps)  # 25

                                data["videos"]["word"].append(word)
                                data["videos"]["word_id"].append(words_to_id[word])

                                data["videos"]["split"].append(set_dict[s])
                                name = os.path.join(
                                    s, word, os.path.basename(output_filename)
                                )
                                data["videos"]["name"].append(name)

                                data["videos"]["mouthing_time"].append(
                                    all_data[s][word]["times"][i]
                                )
                                data["videos"]["mouthing_prob"].append(
                                    all_data[s][word]["probs"][i]
                                )

                                cnt += 1
                    else:
                        raise ValueError(f"Expected completed confidence in labels!")

    print(f"Writing results to {info_file}")
    pkl.dump(data, open(info_file, "wb"))


if __name__ == "__main__":
    description = "Helper script for creating metadata for annotated bsl1k"
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--anno_pkl_path",
        type=Path,
        default="data/bsl1k_annotated/info/annotations_in_mouthing_format_2020.03.04.pkl",
        help="Location of the pkl file containing the human verified annotations.",
    )
    p.add_argument("--data_dir", default="data/bbcsl_annotated", type=Path)
    p.add_argument(
        "--video_folder",
        default="annotated-videos-fixed",
        choices=["annotated-videos-fixed", "annotated-videos", "videos"],
    )
    p.add_argument(
        "--prob_thres",
        type=float,
        default=1.0,
        help="Threshold for the mouthing probability.",
    )
    p.add_argument(
        "-f",
        "--trim-format",
        type=str,
        default="%06d",
        help=(
            "This will be the format for the "
            "filename of trimmed videos: "
            "videoid_%0xd(start_time)_%0xd(end_time).mp4"
        ),
    )
    p.add_argument(
        "--canonical_1064_words",
        type=Path,
        default=("bsldict/subtitles/data/words_mouthing0.8_1064_20.02.21.pkl"),
    )
    main(**vars(p.parse_args()))
