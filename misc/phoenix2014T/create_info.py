"""Create metadata for the phoenix2014T dataset.
"""
import os
import csv
import pickle as pkl
import argparse
from pathlib import Path
from datetime import date

import cv2


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
            # print(video_res_t, count)
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


def main(data_path: Path):
    today_str = date.today().strftime("%y.%m.%d")
    info_file = data_path / "info" / f"info_{today_str}.pkl"
    dict_file = open(data_path / "info" / f"words_{today_str}.txt", "w")

    data = {}
    words = set()
    data["videos"] = {}
    data["videos"]["name"] = []  # Our naming convention (unique ID for a video)
    data["videos"]["glosses"] = []
    data["videos"]["gloss_ids"] = []
    data["videos"]["signer"] = []
    data["videos"]["gloss_seq"] = []
    data["videos"]["sentence"] = []
    data["videos"]["split"] = []  # 0: train, 1: val, 2: test

    # Resolution info
    data["videos"]["videos"] = {}
    data["videos"]["videos"]["T"] = []
    data["videos"]["videos"]["W"] = []
    data["videos"]["videos"]["H"] = []
    data["videos"]["videos"]["duration_sec"] = []
    data["videos"]["videos"]["fps"] = []

    set_dict = {"train": 0, "dev": 1, "test": 2}
    annot_path = "annotations/manual"
    sets = ["train", "dev", "test"]
    sets = {"train": "train-complex-annotation", "dev": "dev", "test": "test"}

    # Read all the sentences from the csvs to determine the vocab for each split
    all_glosses = {}
    for s, v in sets.items():
        annot_file = os.path.join(
            data_path, annot_path, f"PHOENIX-2014-T.{v}.corpus.csv"
        )
        with open(annot_file, newline="") as f:
            reader = csv.reader(f)
            # Skip the header
            next(reader)
            all_glosses[s] = set()
            for row in reader:
                # 2 and 3 are always -1
                gloss_seq = row[0].split("|")[5]
                # sentence = row[0].split('|')[6]
                for g in gloss_seq.split(" "):
                    all_glosses[s].add(g)

    # Check that dev and test words are present in training:
    # 12 words are not
    print([w for w in all_glosses["dev"] if w not in all_glosses["train"]])
    # 19 words are not
    print([w for w in all_glosses["test"] if w not in all_glosses["train"]])

    # Training has 1085 / 1232 (simple / complex)
    words = list(sorted(all_glosses["train"]))
    # The combination has 1115 / 1266 words
    # words = list(
    #     sorted(
    #         all_glosses["train"].union(all_glosses["dev"]).union(all_glosses["test"])
    #     )
    # )

    # Write to TXT file
    words_to_id = {}
    for i, w in enumerate(words):
        words_to_id[w] = i
        dict_file.write(f"{i:05d} {w}\n")
    dict_file.close()

    data["words"] = words
    data["words_to_id"] = words_to_id

    for s, v in sets.items():
        annot_file = os.path.join(
            data_path, annot_path, f"PHOENIX-2014-T.{v}.corpus.csv"
        )
        videos_set = os.path.join(data_path, "videos", s)
        with open(annot_file, newline="") as f:
            reader = csv.reader(f)
            # Skip the header
            next(reader)
            rows = []
            for row in reader:
                rows.append(row[0].split("|"))
        # Loop over annotation csv
        for row in rows:
            # e.g. row[1] is: '01April_2010_Thursday_heute_default-0/1/*.png'
            # Trim the last 8 bits: '01April_2010_Thursday_heute_default-0'
            # This should be our naming convention with gather_frames.py script.
            v = row[1][:-8]
            v = f"{v}.mp4"  # append .mp4
            mp4_path = os.path.join(videos_set, v)
            assert os.path.exists(mp4_path)
            # Video resolution information
            (
                video_res_t,
                video_res_w,
                video_res_h,
                video_fps,
                video_duration_sec,
            ) = _get_video_info(mp4_path)
            # Indication that the video is readable
            if video_res_t:
                data["videos"]["videos"]["T"].append(video_res_t)
                data["videos"]["videos"]["W"].append(video_res_w)  # 210
                data["videos"]["videos"]["H"].append(video_res_h)  # 260
                data["videos"]["videos"]["duration_sec"].append(video_duration_sec)
                data["videos"]["videos"]["fps"].append(video_fps)  # 25

                # E.g. '__ON__ LIEB ZUSCHAUER ABEND WINTER GESTERN loc-NORD SCHOTTLAND
                # loc-REGION UEBERSCHWEMMUNG AMERIKA IX'
                gloss_seq = row[5]
                data["videos"]["sentence"].append(row[6])
                data["videos"]["gloss_seq"].append(gloss_seq)
                glosses = []
                gloss_ids = []
                for g in gloss_seq.split(" "):
                    glosses.append(g)
                    if g in words_to_id:
                        gloss_ids.append(words_to_id[g])
                    else:
                        gloss_ids.append(-1)

                data["videos"]["glosses"].append(glosses)
                data["videos"]["gloss_ids"].append(gloss_ids)

                data["videos"]["split"].append(set_dict[s])
                name = os.path.join(s, v)
                data["videos"]["name"].append(name)
                data["videos"]["signer"].append(row[4])
            else:
                print(mp4_path)

    pkl.dump(data, open(info_file, "wb"))


if __name__ == "__main__":
    description = "Helper script to create info.pkl for Phoenix data."
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--data_path",
        type=Path,
        default="data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T",
        help="Path to Phoenix data.",
    )
    main(**vars(p.parse_args()))
