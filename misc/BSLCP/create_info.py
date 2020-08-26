"""Script to create info file for the BSLCP dataset.

Example usage:
    ipython misc/BSLCP/create_info.py
"""
import argparse
import os
import pickle as pkl
import time
from pathlib import Path


from zsvision.zs_utils import memcache
from beartype import beartype

from misc.bsl1k.extract_clips import construct_video_filename
from misc.BSLCP.extract_clips import time2tuple
from misc.bsl1k_annotated.create_info import _get_video_info


@beartype
def main(
    data_dir: Path,
    json_anno_path: Path,
    video_dir: Path,
    word_data_pkl: Path,
    trim_format: str,
    anno_name: str,
    refresh: bool,
):
    print(f"Creating info file for {anno_name} annotations")
    info_dict_dir = data_dir / "info" / anno_name
    info_dict_dir.mkdir(exist_ok=True, parents=True)
    info_file = info_dict_dir / "info.pkl"
    if info_file.exists() and not refresh:
        print("Found existing info file")
        if word_data_pkl.exists() and not refresh:
            print("Found existing word_data_pkl file")
        else:
            info = memcache(info_file)
            word_data_pkl_data = {key: info[key] for key in ("words", "words_to_id")}
            with open(word_data_pkl, "wb") as f:
                pkl.dump(word_data_pkl_data, f)
            print(f"Wrote word_data_pkl to {word_data_pkl}")
        return
    dict_file = open(info_dict_dir / "words.txt", "w")

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
    data["videos"]["start"] = []
    data["videos"]["end"] = []

    sets = ["train", "val", "test"]
    set_dict = {"train": 0, "val": 1, "test": 2}

    all_data = memcache(json_anno_path)

    words = set()
    for subset, subdict in all_data.items():
        words.update(subdict.keys())

    # Only use train words from reference
    print(f"{len(words)} words")
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
                N = len(all_data[s][word]["start"])
                for i in range(N):
                    start_time = all_data[s][word]["start"][i]
                    end_time = all_data[s][word]["end"][i]
                    output_filename = construct_video_filename(
                        word=word,
                        set_name=s,
                        output_dir=video_dir,
                        name=Path(all_data[s][word]["video"][i]).stem,
                        start_time=time2tuple(start_time),
                        end_time=time2tuple(end_time),
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
                        ) = _get_video_info(str(output_filename))
                        # Indication that the video is readable
                        if video_res_t:
                            # if not (video_fps == row['fps']):
                            #     print(s, i, video_fps, row['fps'])
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

                            data["videos"]["start"].append(
                                all_data[s][word]["start"][i]
                            )
                            data["videos"]["end"].append(
                                all_data[s][word]["end"][i]
                            )
                            cnt += 1
    print(f"Writing results to {info_file}")
    pkl.dump(data, open(info_file, "wb"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anno_name", default="bsl1k_vocab",
                   choices=["bsl1k_vocab", "BSLCP_all_glosses", "signdict_signbank"])
    p.add_argument("--config", type=Path, default="misc/BSLCP/data_paths.json")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    kwargs = vars(args)
    config = memcache(kwargs.pop("config"))
    kwargs.update({
        "json_anno_path": Path(config[args.anno_name]["anno_path"]),
        "trim_format": config["trim_format"],
        "video_dir": Path(config[args.anno_name]["video_dir"]),
        "data_dir": Path(config["data_dir"]),
        "word_data_pkl": Path(config[args.anno_name]["word_data_pkl"]),
    })
    main(**kwargs)
