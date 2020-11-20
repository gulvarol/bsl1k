"""Generate info pickle for the BBCSL dataset.

E.g.
    python misc/bbcsl/create_info.py --mouthing_window_secs=8 --data_dir=data/bbcsl

Or for parallel processing
    python misc/bbcsl/create_info.py --data_dir=data/bbcsl --yaspify --processes 20


"""
import argparse
import json
import multiprocessing as mp
import os
import pickle as pkl
import socket
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tqdm
from zsvision.zs_beartype import beartype
from zsvision.zs_multiproc import starmap_with_kwargs
from misc.bbcsl.extract_clips import load_british_mouthings
from zsvision.zs_utils import memcache

from misc.bbcsl.extract_clips import (
    take_interval_from_peak,
    construct_video_filename,
    pseudo_annos_to_subset_dict,
    get_episode2subset_map,
)


@beartype
def _get_video_info(rgb_path: Path):
    if rgb_path.exists():
        # opencv requires a string input path, rather than Path
        cap = cv2.VideoCapture(str(rgb_path))
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


def update_meta(output_filename, progress_markers, processes, total, count, **kwargs):
    (
        video_res_t,
        video_res_w,
        video_res_h,
        video_fps,
        video_duration_sec,
    ) = _get_video_info(output_filename)

    progress_interval = int(max(total, progress_markers) / progress_markers)

    if processes > 1 and count % progress_interval == 0:
        pct = progress_markers * count / total
        print(f"processing {count}/{total} [{pct:.1f}%] [{output_filename}]")

    if video_res_t:
        kwargs["video_res_t"] = video_res_t
        kwargs["video_res_w"] = video_res_w
        kwargs["video_res_h"] = video_res_h
        kwargs["video_fps"] = video_fps
        kwargs["video_duration_sec"] = video_duration_sec
        kwargs["count"] = count
    else:
        kwargs = None
    return kwargs


@beartype
def filter_words_by_confidence(all_data: Dict, prob_thres: float) -> Dict:
    """Filter each subset of words to include only those with a certain mouthing
    confidence.

    Args:
        all_data (dict): This takes the format
            {"train": {"word1": {"names": [x,x,x], "times": [x,..], "probs"]..}, ..}..}
        prob_thres (float): The minimum required mouthing confidence

    Returns:
        (dict): the filtered data
    """
    subset_words = {}
    for subset in {"train", "val", "test"}:
        keywords = sorted(all_data[subset].keys())
        keep = []
        for keyword in keywords:
            keep_word = (
                np.array(all_data[subset][keyword]["probs"]) > prob_thres
            ).sum()
            keep.append(keep_word > 0)
        print(
            f"Keeping {sum(keep)}/{len(keep)} words from {subset} [thr: {prob_thres}]"
        )
        subset_words[subset] = set(np.array(keywords)[keep].tolist())

    for subset in {"val", "test"}:
        not_in_train = [
            w for w in subset_words[subset] if w not in subset_words["train"]
        ]
        if not_in_train:
            print(
                f"{subset}: removing {not_in_train} [{len(not_in_train)} not in train]"
            )
        subset_words[subset] = subset_words[subset] - set(not_in_train)
        # Expected -> okay, sixty

    for subset in {"train", "val", "test"}:
        keep = subset_words[subset]
        all_data[subset] = {
            keyword: val for keyword, val in all_data[subset].items() if keyword in keep
        }
        print(f"Keeping {len(all_data[subset])} words from {subset} after filtering")
    return all_data


@beartype
def enforce_1064_vocab(
    all_data: Dict[str, Dict], canonical_vocab: set,
) -> Dict[str, Dict]:
    """Checks that the data contains the required 1064 vocbulary list and filters
    so that no extraneous words are included.
    """
    assert len(canonical_vocab) == 1064, "unexpected number of words"
    for subset, subdict in all_data.items():
        prev_len = len(subdict)
        all_data[subset] = {
            key: val for key, val in all_data[subset].items() if key in canonical_vocab
        }
        print(f"Filtered {subset} from {prev_len} to {len(all_data[subset])}")
    return all_data


@beartype
def gen_paths(
    data_dir: Path,
    prob_thres: float,
    pseudo_annos: str,
    worker_id: int,
    num_partitions: int,
    limit: int,
    mouthing_window_secs: int,
) -> Dict[str, Path]:
    today_str = date.today().strftime("%y.%m.%d")
    info_fname = f"info_{today_str}-thr{prob_thres}"
    word_fname = f"words_{today_str}-thr{prob_thres}"
    tag = ""
    if mouthing_window_secs:
        # update the new components to use the signhd suffix to indicate that the
        # correct source videos are used.
        tag += f"-{mouthing_window_secs}sec-window-signhd"
    if limit:
        tag += f"-limit-{limit}"
    if pseudo_annos:
        tag += f"-pseudo-annos-{pseudo_annos}"

    info_fname += tag
    word_fname += tag
    paths = {
        "info": f"{info_fname}.pkl",
        "words": f"{word_fname}.txt",
    }
    paths = {key: data_dir / "info" / val for key, val in paths.items()}
    # create new locations for parallel files
    if num_partitions > 1:
        for key in {"words", "info"}:
            path = paths[key]
            tag = f"-{worker_id:02d}-{num_partitions:02d}"
            fname = f"{path.stem}{tag}{path.suffix}"
            paths[key] = path.parent / "info-partitions" / fname

    print("Generated paths")
    for key, val in paths.items():
        print(f"{key} -> {val}")
        # ensure parents exist for new files
        if key in {"info", "words"}:
            val.parent.mkdir(exist_ok=True, parents=True)
    return paths


@beartype
def create_info_structure() -> Dict[str, Dict]:
    data = {}
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
    return data


@beartype
def load_data(
    anno_pkl_path: Path,
    canonical_vocab: set,
    pseudo_annos: str,
    episode2subset: Dict[str, str],
) -> Dict:
    if "mouthing" in str(anno_pkl_path):
        msg = "Pseudo anno type should not be specified for mouthing annos"
        assert not pseudo_annos, msg
        all_data = load_british_mouthings(anno_pkl_path)
        all_data = enforce_1064_vocab(all_data, canonical_vocab)
    elif "pseudo-annos" in str(anno_pkl_path):
        all_data = pseudo_annos_to_subset_dict(
            pseudo_anno_path=anno_pkl_path,
            canonical_vocab=canonical_vocab,
            pseudo_annos=pseudo_annos,
            episode2subset=episode2subset,
        )
    else:
        raise ValueError(f"Unsupported anno path: {anno_pkl_path}")
    return all_data


@beartype
def main(
    data_dir: Path,
    anno_pkl_path: Path,
    video_dir: Path,
    canonical_1064_words: Path,
    refresh: bool,
    prob_thres: float,
    worker_id: int,
    num_partitions: int,
    limit: int,
    processes: int,
    mouthing_window_secs: int,
    progress_markers: int,
    aggregate: bool,
    pseudo_annos: str,
    episode2subset: Dict[str, str],
    trim_format: str = "%06d",
):
    path_kwargs = {
        "limit": limit,
        "data_dir": data_dir,
        "pseudo_annos": pseudo_annos,
        "prob_thres": prob_thres,
        "mouthing_window_secs": mouthing_window_secs,
    }
    with open(canonical_1064_words, "rb") as f:
        canonical_vocab = set(pkl.load(f)["words"])

    if aggregate:
        dest_path = gen_paths(worker_id=0, num_partitions=1, **path_kwargs)["info"]
        if dest_path.exists() and not refresh:
            print(f"Found existing info file at {dest_path}, skipping...")
            return
        info = create_info_structure()
        for ii in range(num_partitions):
            src_path = gen_paths(
                worker_id=ii, num_partitions=num_partitions, **path_kwargs
            )["info"]
            worker_info = memcache(src_path)
            msg = "Expected worker info to match the target 1064 vocab"
            assert set(worker_info["words"]) == canonical_vocab, msg
            if ii == 0:
                # we can update the words with the first worker
                info["words"] = worker_info["words"]
                info["words_to_id"] = worker_info["words_to_id"]
            for key in info["videos"]:
                if key == "videos":
                    for subkey in info["videos"]["videos"]:
                        info["videos"]["videos"][subkey].extend(
                            worker_info["videos"]["videos"][subkey]
                        )
                else:
                    info["videos"][key].extend(worker_info["videos"][key])
        print(f"Writing aggregated info to {dest_path}")
        with open(dest_path, "wb") as f:
            pkl.dump(info, f)
        return

    paths = gen_paths(worker_id=worker_id, num_partitions=num_partitions, **path_kwargs)
    if paths["info"].exists() and not refresh:
        print(f"Found existing info file at {paths['info']}, skipping...")
        return

    data = create_info_structure()
    words = set()
    sets = ["train", "val", "test"]
    set_dict = {"train": 0, "val": 1, "test": 2}
    all_data = load_data(
        pseudo_annos=pseudo_annos,
        anno_pkl_path=anno_pkl_path,
        canonical_vocab=canonical_vocab,
        episode2subset=episode2subset,
    )
    all_data = filter_words_by_confidence(all_data, prob_thres)
    print(f"Using a vocabulary of {len(canonical_vocab)} words for BBC")
    words = list(sorted(canonical_vocab))

    # Write to TXT file
    with open(paths["words"], "w") as dict_file:
        words_to_id = {}
        for i, w in enumerate(words):
            words_to_id[w] = i
            dict_file.write(f"{i:05d} {w}\n")

    data["words"] = words
    data["words_to_id"] = words_to_id

    t0 = time.time()
    if num_partitions == 1:
        worker_words = set(words)
    else:
        worker_words = np.array_split(words, num_partitions)[worker_id]

    count = 0
    kwarg_list = []
    for s in sets:  # all_data.keys():
        subset_total = len(all_data[s])
        for word_cnt, word in enumerate(all_data[s].keys()):
            assert word in words_to_id, f"Unkown word: {word}"
            if limit and count >= limit:
                continue
            if word not in worker_words:
                continue
            N = len(all_data[s][word]["names"])
            delta = time.time() - t0
            print(
                f"{delta:0.2f} sec {s} {word_cnt}/{subset_total} {word} [{N} samples]"
            )
            for i in range(N):
                if all_data[s][word]["probs"][i] > prob_thres:
                    start_time, end_time = take_interval_from_peak(
                        all_data[s][word]["times"][i]
                    )
                    output_filename = construct_video_filename(
                        output_dir=video_dir,
                        set_name=s,
                        word=word,
                        name=all_data[s][word]["names"][i],
                        start_time=start_time,
                        end_time=end_time,
                        trim_format=trim_format,
                    )
                    if os.path.exists(output_filename):
                        # Video resolution information
                        name = os.path.join(s, word, os.path.basename(output_filename))
                        kwargs = {
                            "count": count,
                            "word": word,
                            "name": name,
                            "word_id": words_to_id[word],
                            "split": set_dict[s],
                            "processes": processes,
                            "mouthing_time": all_data[s][word]["times"][i],
                            "mouthing_prob": all_data[s][word]["probs"][i],
                            "output_filename": output_filename,
                            "progress_markers": progress_markers,
                        }
                        kwarg_list.append(kwargs)
                        count += 1

    # Enable the worker to print progress.
    for kwargs in kwarg_list:
        kwargs["total"] = len(kwarg_list)

    func = update_meta
    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            meta = starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)
    else:
        meta = []
        for kwargs in tqdm.tqdm(kwarg_list):
            meta.append(func(**kwargs))

    # Filter videos that failed to return meta data
    pre_filter = len(meta)
    meta = [x for x in meta if x]
    print(f"{len(meta)}/{pre_filter} were successfully parsed for meta information")

    # check that ordering was preserved by multiprocessing
    counts = [x["count"] for x in meta]
    assert list(sorted(counts)) == counts, "Expected meta items to be in order"

    for x in tqdm.tqdm(meta):
        data["videos"]["videos"]["T"].append(x["video_res_t"])
        data["videos"]["videos"]["W"].append(x["video_res_w"])  # 480
        data["videos"]["videos"]["H"].append(x["video_res_h"])  # 480
        data["videos"]["videos"]["duration_sec"].append(x["video_duration_sec"])
        data["videos"]["videos"]["fps"].append(x["video_fps"])  # 25
        data["videos"]["word"].append(x["word"])
        data["videos"]["word_id"].append(x["word_id"])
        data["videos"]["split"].append(x["split"])
        data["videos"]["name"].append(x["name"])
        data["videos"]["mouthing_time"].append(x["mouthing_time"])
        data["videos"]["mouthing_prob"].append(x["mouthing_prob"])
    print(f"Saving info file to {paths['info']}...")
    pkl.dump(data, open(paths["info"], "wb"))


if __name__ == "__main__":
    description = "Helper script to create info.pkl for bbcsl data."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--progress_markers", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_partitions", default=1, type=int)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="/users/gul/datasets/bbcsl",
        help="Directory where videos are.",
    )
    parser.add_argument(
        "--prob_thres",
        type=float,
        default=0.5,
        help="Threshold for the mouthing probability.",
    )
    parser.add_argument(
        "--mouthing_window_secs",
        default=8,
        type=int,
        help="if given, preprocess videos from different windows.",
    )
    parser.add_argument("--worker_id", default=0, type=int)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--yaspify", action="store_true")
    parser.add_argument("--run_on_gnodes", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--constraint_str", default="", help="slurm constraints")
    parser.add_argument(
        "--canonical_1064_words",
        type=Path,
        default="bsldict/subtitles/data/words_mouthing0.8_1064_20.02.21.pkl",
    )
    parser.add_argument("--yaspi_defaults_path", default="misc/yaspi_cpu_defaults.json")
    parser.add_argument(
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
    parser.add_argument(
        "--anno_pkl_path",
        type=Path,
        required=True,
        choices=[
            Path("data/bbcsl/info/mouthings_2020.02.12.pkl"),  # original
        ],
    )
    parser.add_argument(
        "--subset2episode", type=Path, default="data/bbcsl_raw/subset2episode.pkl"
    )
    parser.add_argument(
        "--pseudo_annos",
        default="",
        choices=["raw-boost-only", "cls-freq", "raw", "mouthing"],
        help="if given, use this kind of pseudo annotation",
    )
    parser.add_argument("--video_dir", type=Path, required=True)
    args = parser.parse_args()

    episode2subset = get_episode2subset_map(args.subset2episode)

    if args.yaspify:
        # Only import yaspi if requested
        from yaspi.yaspi import Yaspi

        with open(args.yaspi_defaults_path, "r") as f:
            yaspi_defaults = json.load(f)
        cmd_args = sys.argv
        cmd_args.remove("--yaspify")
        base_cmd = f"python {' '.join(cmd_args)}"
        job_name = f"create-{args.num_partitions}-info-windows"
        yaspi_defaults["constraint_str"] = args.constraint_str
        yaspi_defaults["partition"] = "gpu" if args.run_on_gnodes else "compute"
        job = Yaspi(
            cmd=base_cmd,
            job_queue=None,
            gpus_per_task=0,
            job_name=job_name,
            job_array_size=args.num_partitions,
            **yaspi_defaults,
        )
        job.submit(watch=True, conserve_resources=5)
    else:
        if args.slurm:
            if socket.gethostname().endswith("cluster"):
                os.system(str(Path.home() / "configure_tmp_data.sh"))
        main(
            limit=args.limit,
            refresh=args.refresh,
            data_dir=args.data_dir,
            processes=args.processes,
            prob_thres=args.prob_thres,
            trim_format=args.trim_format,
            worker_id=args.worker_id,
            num_partitions=args.num_partitions,
            episode2subset=episode2subset,
            progress_markers=args.progress_markers,
            mouthing_window_secs=args.mouthing_window_secs,
            canonical_1064_words=args.canonical_1064_words,
            aggregate=args.aggregate,
            anno_pkl_path=args.anno_pkl_path,
            pseudo_annos=args.pseudo_annos,
            video_dir=args.video_dir,
        )
