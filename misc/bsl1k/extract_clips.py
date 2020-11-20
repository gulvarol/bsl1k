"""Helper script to extract clips for BSL-1K

Sample usage:
    ipython misc/bsl1k/extract_clips.py
"""
import os
import sys
import json
import pickle as pkl
import shutil
import socket
import argparse
import functools
import subprocess
import multiprocessing as mp
from typing import Dict, List, Tuple, Union
from numbers import Number
from pathlib import Path
from datetime import date
from collections import defaultdict

import tqdm
import numpy as np
from typeguard import typechecked
from frozendict import frozendict
from zsvision.zs_utils import BlockTimer, memcache
from zsvision.zs_beartype import beartype
from zsvision.zs_multiproc import starmap_with_kwargs


US2UK_MAPPING = {
    'airplane': 'aeroplane',
    'center': 'centre',
    'favor': 'favour',
    'gray': 'grey',
    'practice': 'practise',
    'recognize': 'recognise',
    'yogurt': 'yoghurt',
}


def clean_subtitle_word(x):
    drop = ['"', ".", ",", "!", "?"]
    for punc in drop:
        x = x.replace(punc, "")
    if x and x[0] == "'":
        x = x[1:]
    if x and x[-1] == "'":
        x = x[:-1]
    return x.lower()


@beartype
def extract_clip(
    source_file: Path,
    output_filename: Path,
    start_time: float,
    end_time: float,
    force_resize: int,
    refresh: bool = False,
):
    """Extract a clip from a video.

    arguments:
    ---------
    source_file: Path to untrimmed video
    output_filename: File path where the video will be stored.
    start_time: Indicates begining time in seconds from where the video will be trimmed.
    end_time: Indicates ending time in seconds of the trimmed video.
    force_resize: Resize video so both dimensions match this value (breaks aspect ratio).
    """
    if output_filename.exists() and not refresh:
        print(f"Found existing video at {output_filename}, skipping")
        return
    # Ensure that the operation atomic
    tmp_fname = f"{output_filename.stem}-tmpvid.mp4"
    output_filename_tmp = output_filename.parent / tmp_fname
    output_filename_tmp.parent.mkdir(exist_ok=True, parents=True)
    assert force_resize, "For efficiency, expected force_resize to be > 0"

    # Construct command to trim the videos (ffmpeg required).
    flags = (f" -c:v libx264 -crf 18 -pix_fmt yuv420p -g 16"
             f" -profile:v high -vf fps=fps=25").split()
    command = ['ffmpeg', ' -y '
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-i', "'%s'" % source_file,
               '-threads', '1',
               *flags,
               '"%s"' % output_filename_tmp]

    if force_resize:
        command.insert(7, f"-vf scale={force_resize}:{force_resize}")
    command = " ".join(command)
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        shutil.move(output_filename_tmp, output_filename)
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        print(f"Failed to extract {output_filename_tmp} [{err}]")


@beartype
def pseudo_annos_to_subset_dict(
    pseudo_anno_path: Path,
    pseudo_annos: str,
    canonical_vocab: set,
    episode2subset: Dict[str, str],
) -> Dict[str, Dict]:
    # keep track of some basic stats as a sanity check
    thresholds = [0.5, 0.7, 0.9]
    counts = {thr: 0 for thr in thresholds}
    data = memcache(pseudo_anno_path)[pseudo_annos]
    subset_data = {key: dict() for key in ("train", "val", "test")}
    subset2episodes = {key: set() for key in subset_data}
    for episode, subset in episode2subset.items():
        subset2episodes[subset].add(episode)
    for subset in subset_data:
        for word, worddict in tqdm.tqdm(data.items()):
            assert word in canonical_vocab, f"Expected {word} to be in 1064 vocab"
            keep = np.array([x in subset2episodes[subset] for x in worddict["names"]])
            if keep.sum():
                if word not in subset_data[subset]:
                    subset_data[subset][word] = defaultdict(list)
                for key, val in worddict.items():
                    kept = np.array(val)[keep].tolist()
                    subset_data[subset][word][key].extend(kept)
                for thr in counts:
                    counts[thr] += (np.array(worddict["probs"])[keep] > thr).sum()
    data = subset_data
    for thr, val in counts.items():
        print(f"Found {val} annotations at confidences > {thr}")
    return data


@beartype
def get_episode2subset_map(subset2episode: Path) -> Dict[str, str]:
    """Build a mapping that converts episode keys into their respective subsets
    """
    subset2episode = memcache(subset2episode)
    episode2subset = {}
    for subset, episodes in subset2episode.items():
        for episode in episodes:
            episode_key = episode.replace("/", "--")
            assert episode_key not in episode2subset, f"Duplicate key: {episode}!"
            episode2subset[episode_key] = subset
    return episode2subset


@beartype
def take_interval_from_peak(
    time_in_sec: float,
    fps: int = 25,
    num_frames_before: int = 64,
    num_frames_after: int = 0,
) -> Tuple[float, float]:
    padding_before = num_frames_before / fps
    padding_after = num_frames_after / fps
    start_time = time_in_sec - padding_before
    end_time = time_in_sec + padding_after
    return start_time, end_time


@typechecked
def construct_video_filename(
    output_dir: Path,
    set_name: str,
    word: str,
    name: str,
    start_time: Union[float, Tuple[float, int]],
    end_time: Union[float, Tuple[float, int]],
    trim_format: str,
) -> Path:
    output_filename = "%s_%s_%s.mp4" % (
        name,
        trim_format % start_time,
        trim_format % end_time,
    )
    return output_dir / set_name / word / output_filename


@beartype
def construct_original_filename(name: str, video_src_name: str) -> Path:
    # videos_parent_orig = '/scratch/shared/beegfs/albanie/exp/kws-align/bsl1k/videos-mp4'
    video_parent_orig = (
        "/scratch/shared/beegfs/albanie/shared-datasets/bsl1k/videos-mp4/"
    )
    parent_folder, base_folder = name.split("--")
    fname = f"{video_src_name}.mp4"
    return Path(video_parent_orig) / parent_folder / base_folder / fname


@beartype
def load_british_mouthings(mouthing_pkl_path: Path) -> dict:
    """Load mouthing predictions from disk and transform the keywords from US to UK
    English.
    """
    # Note: we leave the practice/practise dilemena for another time and stick with this
    # list for backwards compatibility
    us_mouthings = memcache(mouthing_pkl_path)
    british_mouthings = {}
    for subset, subdict in us_mouthings.items():
        british_mouthings[subset] = {US2UK_MAPPING.get(key, key): val
                                     for key, val in subdict.items()}
    return british_mouthings


@beartype
@functools.lru_cache(maxsize=64, typed=False)
def parse_subtitles(
    subtitle_pkl_path: Path,
    subtitle_reference_mouthings: Path,
    canonical_vocab: frozenset,
    prob_thres: Number,
    episode2subset: frozendict,
    pkl_file: Path = None,
    episode_filter: str = None,
    save_pkl: bool = True,
    temporal_tol: int = 4,
) -> Dict:
    """Extract raw subtitles into a format that mimics the mouthing predictions. Use
    frozen datastructures to allow LRU caching.
    """
    subs = memcache(subtitle_pkl_path)
    ref_mouthings = load_british_mouthings(subtitle_reference_mouthings)

    # Filter to episodes with available subtitles
    subset2episodes = defaultdict(list)
    for episode, subset in episode2subset.items():
        episode = episode.replace("/", "--")
        if episode_filter and episode_filter not in episode:
            continue
        if episode in subs:
            subset2episodes[subset].append(episode)
    print(f"Filtered to {sum(len(x) for x in subset2episodes.values())} episodes")

    data = {}
    count = 0
    for subset, episodes in subset2episodes.items():
        data[subset] = {}
        for episode in tqdm.tqdm(episodes):
            episode_subs = subs[episode]
            for sub in tqdm.tqdm(episode_subs):
                if isinstance(sub["span"], list):
                    text = "".join([x["text"] for x in sub["span"]])
                else:
                    text = sub["span"]["text"]
                subtitle_words = [clean_subtitle_word(x) for x in text.split(" ")]
                for keyword in canonical_vocab:
                    keyword_ref_mouthings = ref_mouthings[subset][keyword]
                    keep = keyword_ref_mouthings["names"] == episode
                    conf_keep = np.array(keyword_ref_mouthings["probs"]) > prob_thres
                    mask = conf_keep * keep
                    if prob_thres and not (keep.sum() and mask.sum()):
                        continue
                    candidate_times = np.array(keyword_ref_mouthings["times"])[mask]

                    if keyword not in data[subset]:
                        data[subset][keyword] = {"names": [], "probs": [], "times": []}
                    if keyword in subtitle_words:
                        sub_time = sub["start"] + (sub["end"] - sub["start"]) / 2
                        candidate_times = np.array(keyword_ref_mouthings["times"])[mask]
                        if prob_thres:
                            # we only keep times that are close to a confident mouthing
                            if np.abs(candidate_times - sub_time).min() > temporal_tol:
                                continue

                        data[subset][keyword]["names"].append(episode)
                        data[subset][keyword]["probs"].append(1)
                        data[subset][keyword]["times"].append(sub_time)
                        count += 1
    print(f"Proposing {count} subtitle crops")
    if save_pkl:
        pkl.dump(data, open(pkl_file, "wb"))
    return data


@beartype
def gather_all_jsons(
    pkl_file: Path, window_secs: int, save_pkl=True
) -> Dict[str, Dict]:
    # JSON files of mouthing outputs for 1.3K words
    base = Path("/scratch/shared/beegfs/albanie/exp/kws-align/saved/log")
    dataset = "bbc-1k3-eval"
    subsets = {"train", "val", "test"}
    dataset = f"{dataset}-{window_secs}sec-window"
    sets = {key: base / dataset / f"no-timestamp/{key}-nms-0.6" for key in subsets}
    data = {}
    for s, set_path in tqdm.tqdm(sets.items()):
        data[s] = {}
        for json_file in tqdm.tqdm(os.listdir(set_path)):
            print(json_file)
            word = json_file[:-5]
            data[s][word] = {}
            json_path = os.path.join(set_path, json_file)
            with open(json_path) as ipf:
                content = json.load(ipf)
                data[s][word]["names"] = content["names"]
                data[s][word]["times"] = content["times"]
                data[s][word]["probs"] = content["probs"]
    if save_pkl:
        pkl.dump(data, open(pkl_file, "wb"))
    return data


@beartype
def gen_paths(
    output_dir: Path,
    pseudo_anno_path: Path,
    window_secs: int,
    use_date: str,
    limit: int,
    video_src_name: str,
    use_subs: bool,
    use_sentences: bool,
    pseudo_annos: str,
    prob_thres: float,
    force_resize: int,
    num_frames_before: int,
    num_frames_after: int,
) -> Dict[str, Path]:
    if use_subs:
        fname = f"subtitle-timings-mimic-thr-{prob_thres}"
        msg = "Expected subtitle clips to be centered (equal frames before and after)"
        assert num_frames_before == num_frames_after, msg
        msg = "Expected subtitle clips to be 64 frames long"
        assert num_frames_before == 32, msg
        assert num_frames_after == 32, msg
    else:
        # Set the date string (either custom, or today)
        if use_date:
            date_str = use_date
        else:
            date_str = date.today().strftime("%y.%m.%d")
        # Annotation pkl to use
        if pseudo_annos:
            fname = f"pseudo_{date_str}"
        elif use_sentences:
            fname = f"subtitles_{date_str}"
        else:
            fname = f"mouthings_{date_str}"
            if window_secs:
                fname += f"-{window_secs}sec-window"

    if not use_subs:
        msg = "Expected mouthing clips to be 64 frames long, and end at the peak"
        assert num_frames_before == 64, msg
        assert num_frames_after == 0, msg

    # Add tags to uniquely idenitfy the output
    tag = ""
    if force_resize:
        tag += f"-resized-25fps-{force_resize}x{force_resize}"
    if window_secs:
        tag += f"-{window_secs}sec-window"
    if use_subs:
        tag += f"-subtitles"
    elif pseudo_annos:
        tag += f"-pseudo-annos-{pseudo_annos}-{pseudo_anno_path.parent.stem}"
    elif use_sentences:
        tag += f"-sentences"
    if limit:
        tag += f"-limit-{limit}"
    paths = {
        "anno_pkl": Path("info") / f"{fname}.pkl",
        "kwargs_pkl": Path("processing") / "kwargs.pkl",
        "output_dir_videos": Path("video_dirs") / f"videos-{video_src_name}",
    }
    for key, rel_path in paths.items():
        # TODO: Check with Sam.
        if key == "anno_pkl":
            paths[key] = output_dir / rel_path
        else:
            suffix = rel_path.suffix
            paths[key] = output_dir / rel_path.parent / f"{rel_path.stem}{tag}{suffix}"
    print(f"Generated paths:")
    for key, path in paths.items():
        print(f"{key} -> {path}")
    return paths


@beartype
def build_kwarg_list_for_word(
    refresh: bool,
    word: str,
    subset: str,
    trim_format: str,
    video_src_name: str,
    prob_thres: float,
    num_frames_before: int,
    num_frames_after: int,
    count: int,
    total: int,
    processes: int,
    force_resize: int,
    word_data: Dict,
    paths: Dict[str, Path],
) -> List[Dict]:
    kwarg_list = []
    N = len(word_data["names"])

    # show rudimentary estimate of current progress when using multiprocessing
    progress_markers = 100
    progress_interval = int(max(total, progress_markers) / progress_markers)
    if processes > 1 and count % progress_interval == 0:
        pct = progress_markers * count / total
        print(f"Processing [{subset} {word}] [{pct:.1f}%] item {count}/{total}")

    for i in range(N):
        if word_data["probs"][i] > prob_thres:
            start_time, end_time = take_interval_from_peak(
                time_in_sec=word_data["times"][i],
                num_frames_before=num_frames_before,
                num_frames_after=num_frames_after,
            )
            output_filename = construct_video_filename(
                output_dir=paths["output_dir_videos"],
                set_name=subset,
                word=word,
                name=word_data["names"][i],
                start_time=start_time,
                end_time=end_time,
                trim_format=trim_format,
            )
            if not Path(output_filename).exists() or refresh:
                source_file = construct_original_filename(
                    name=word_data["names"][i], video_src_name=video_src_name,
                )
                kwargs = {
                    "source_file": source_file,
                    "force_resize": force_resize,
                    "output_filename": output_filename,
                    "start_time": start_time,
                    "end_time": end_time,
                }
                kwarg_list.append(kwargs)
    return kwarg_list


@beartype
def build_kwarg_list_for_sentence(
    refresh: bool,
    subset: str,
    trim_format: str,
    video_src_name: str,
    force_resize: int,
    subtitle_data: Dict,
    paths: Dict[str, Path],
    sentence_pad_sec: int,
    total: int,
) -> List[Dict]:
    kwarg_list = []
    # subtitle_data for this split has
    # 'subtitle', 'subtitle_lemmas', 'name', 'start_time', 'end_time'
    # N = number of sentences in split (all episodes combined) e.g. 675K
    N = len(subtitle_data['start_time'])

    for i in tqdm.tqdm(range(N)):
        start_time = subtitle_data['start_time'][i] - sentence_pad_sec
        end_time = subtitle_data['end_time'][i] + sentence_pad_sec
        episode_name = subtitle_data['name'][i]
        output_filename = construct_video_filename(
            output_dir=paths["output_dir_videos"],
            set_name=subset,
            word=episode_name,  # group per episode
            name=episode_name,
            start_time=start_time,
            end_time=end_time,
            trim_format=trim_format,
        )
        if not Path(output_filename).exists() or refresh:
            source_file = construct_original_filename(
                name=episode_name, video_src_name=video_src_name,
            )
            kwargs = {
                "source_file": source_file,
                "force_resize": force_resize,
                "output_filename": output_filename,
                "start_time": start_time,
                "end_time": end_time,
            }
            kwarg_list.append(kwargs)
    return kwarg_list


@beartype
def main(
    output_dir: Path,
    subtitle_pkl_path: Path,
    canonical_1064_words: Path,
    pseudo_anno_path: Path,
    subtitle_reference_mouthings: Path,
    use_date: str,
    trim_format: str,
    video_src_name: str,
    refresh: bool,
    refresh_kwargs_pkl: bool,
    use_subs: bool,
    use_sentences: bool,
    pseudo_annos: str,
    kwargs_only: bool,
    limit: int,
    worker_id: int,
    processes: int,
    force_resize: int,
    num_partitions: int,
    num_frames_before: int,
    num_frames_after: int,
    window_secs: int,
    prob_thres: float,
    episode2subset: Dict[str, str],
):

    paths = gen_paths(
        limit=limit,
        use_date=use_date,
        video_src_name=video_src_name,
        output_dir=output_dir,
        use_subs=use_subs,
        use_sentences=use_sentences,
        pseudo_annos=pseudo_annos,
        pseudo_anno_path=pseudo_anno_path,
        prob_thres=prob_thres,
        force_resize=force_resize,
        num_frames_before=num_frames_before,
        num_frames_after=num_frames_after,
        window_secs=window_secs,
    )
    with open(canonical_1064_words, "rb") as f:
        canonical_vocab = set(pkl.load(f)["words"])
    if pseudo_annos:
        data = pseudo_annos_to_subset_dict(
            pseudo_anno_path=pseudo_anno_path,
            pseudo_annos=pseudo_annos,
            episode2subset=episode2subset,
            canonical_vocab=canonical_vocab,
        )
    elif paths["anno_pkl"].exists():
        data = pkl.load(open(paths["anno_pkl"], "rb"))
    else:
        print(f"Generating pkl file for {window_secs} sec window...")
        if use_subs:
            data = parse_subtitles(
                pkl_file=paths["anno_pkl"],
                prob_thres=prob_thres,
                episode2subset=episode2subset,
                canonical_vocab=canonical_vocab,
                subtitle_reference_mouthings=subtitle_reference_mouthings,
                subtitle_pkl_path=subtitle_pkl_path,
            )
        else:
            data = gather_all_jsons(paths["anno_pkl"], window_secs=window_secs)

    if paths["kwargs_pkl"].exists() and not refresh_kwargs_pkl:
        print(f"Loading kwargs from {paths['kwargs_pkl']} from cache")
        with open(paths["kwargs_pkl"], "rb") as f:
            kwarg_list = pkl.load(f)
    else:
        if use_sentences:
        # Parallization doesn't really make sense for sentences,
        # but we keep it to preserve structure.
            count = 0
            kwarg_constructors = []
            for subset in tqdm.tqdm(data.keys()):
                if limit and count >= limit:
                    continue
                kwargs = {
                    "refresh": refresh,
                    "subset": subset,
                    "paths": paths,
                    "subtitle_data": data[subset],
                    "trim_format": trim_format,
                    "video_src_name": video_src_name,
                    "force_resize": force_resize,
                    "sentence_pad_sec": window_secs,
                }
                kwarg_constructors.append(kwargs)
                count += 1
            func = build_kwarg_list_for_sentence
        else:
            # Due to the scale of the preprocessing, the algorithm for creating the
            # arguments that will be passed to each worker is also parallelised
            # (i.e. we are using multiprocessing to determine the keyword arguments)
            count = 0
            kwarg_constructors = []
            for subset in tqdm.tqdm(data.keys()):
                for word in tqdm.tqdm(data[subset].keys()):
                    if limit and count >= limit:
                        continue
                    kwargs = {
                        "refresh": refresh,
                        "word": word,
                        "count": count,
                        "subset": subset,
                        "paths": paths,
                        "prob_thres": prob_thres,
                        "word_data": data[subset][word],
                        "trim_format": trim_format,
                        "video_src_name": video_src_name,
                        "num_frames_before": num_frames_before,
                        "num_frames_after": num_frames_after,
                        "force_resize": force_resize,
                        "processes": processes,
                    }
                    kwarg_constructors.append(kwargs)
                    count += 1
            func = build_kwarg_list_for_word

        # Include total counts to allow the function to show progress
        for kwargs in kwarg_constructors:
            kwargs["total"] = count
        with BlockTimer("Building kwarg lists"):
            if processes > 1:
                with mp.Pool(processes=processes) as pool:
                    kwarg_list = starmap_with_kwargs(
                        pool=pool, func=func, kwargs_iter=kwarg_constructors,
                    )
            else:
                kwarg_list = []
                for kwargs in tqdm.tqdm(kwarg_constructors):
                    kwarg_list.append(func(**kwargs))

        # flatten outputs
        kwarg_list = [x for sublist in kwarg_list for x in sublist]
        print(
            f"Caching kwarg_list ({len(kwarg_list)} elements) to {paths['kwargs_pkl']}"
        )
        paths["kwargs_pkl"].parent.mkdir(exist_ok=True, parents=True)
        with open(paths["kwargs_pkl"], "wb") as f:
            pkl.dump(kwarg_list, f)

    if kwargs_only:
        return

    kwarg_list = np.array_split(kwarg_list, num_partitions)[worker_id]
    msg = (
        f"Worker {worker_id}/{num_partitions} processing {len(kwarg_list)} items"
        f" with {processes} processes"
    )
    print(msg)
    if limit:
        kwarg_list = kwarg_list[:limit]
    func = extract_clip
    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)
    else:
        for kwargs in tqdm.tqdm(kwarg_list):
            func(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_date", default="")
    parser.add_argument("--use_subs", action="store_true")
    parser.add_argument("--use_sentences", action="store_true")
    parser.add_argument(
        "--pseudo_annos",
        default="",
        choices=["raw-boost-only", "cls-freq", "raw", "mouthing"],
        help="if given, use this kind of pseudo annotation",
    )
    parser.add_argument(
        "--pseudo_anno_path",
        type=Path,
        default="data/bsl1k/pseudo-annos/006-nbrs-aggregated.pkl",
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="data/bsl1k",
        help="Output directory where videos will be saved.",
    )
    parser.add_argument(
        "--window_secs", type=int, default=8, help="window second duration"
    )
    parser.add_argument(
        "--prob_thres",
        type=float,
        default=0.5,
        help="Threshold for the mouthing probability.",
    )
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
    parser.add_argument("--yaspify", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--kwargs_only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh_kwargs_pkl", action="store_true")
    parser.add_argument(
        "--video_src_name",
        default="signhd",
        choices=["signhd", "signhd-dense-fast-audio"],
    )
    parser.add_argument("--worker_id", default=0, type=int)
    parser.add_argument("--force_resize", default=256, type=int)
    parser.add_argument(
        "--canonical_1064_words",
        type=Path,
        default="misc/bsldict/subtitles/data/words_mouthing0.8_1064_20.02.21.pkl",
    )
    parser.add_argument("--num_partitions", default=1, type=int)
    parser.add_argument("--yaspi_defaults_path", default="misc/yaspi_cpu_defaults.json")
    parser.add_argument(
        "--subtitle_pkl_path",
        type=Path,
        default="data/bsl1k_raw/processing/aggregated_subs.pkl",
    )
    parser.add_argument(
        "--subset2episode", type=Path, default="data/bsl1k_raw/subset2episode.pkl"
    )
    parser.add_argument(
        "--num_frames_before",
        type=int,
        default=64,
        help="the number of frames to keep before the peak",
    )
    parser.add_argument(
        "--num_frames_after",
        type=int,
        default=0,
        help="the number of frames to keep after the peak",
    )
    parser.add_argument(
        "--subtitle_reference_mouthings",
        type=Path,
        default="data/bsl1k/info/mouthings_20.02.23-8sec-window.pkl",
    )
    args = parser.parse_args()

    if args.slurm and socket.gethostname().endswith("cluster"):
        os.system(str(Path.home() / "configure_tmp_data.sh"))
    episode2subset = get_episode2subset_map(args.subset2episode)

    if args.yaspify:
        # Only import yaspi if requested
        from yaspi.yaspi import Yaspi
        
        with open(args.yaspi_defaults_path, "r") as f:
            yaspi_defaults = json.load(f)
        cmd_args = sys.argv
        cmd_args.remove("--yaspify")
        base_cmd = f"python {' '.join(cmd_args)}"
        job_name = f"extract-clips-{args.num_partitions}-partitions"
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
        main(
            limit=args.limit,
            refresh=args.refresh,
            refresh_kwargs_pkl=args.refresh_kwargs_pkl,
            use_subs=args.use_subs,
            use_sentences=args.use_sentences,
            worker_id=args.worker_id,
            processes=args.processes,
            kwargs_only=args.kwargs_only,
            video_src_name=args.video_src_name,
            window_secs=args.window_secs,
            num_partitions=args.num_partitions,
            output_dir=args.output_dir,
            trim_format=args.trim_format,
            prob_thres=args.prob_thres,
            use_date=args.use_date,
            force_resize=args.force_resize,
            canonical_1064_words=args.canonical_1064_words,
            num_frames_before=args.num_frames_before,
            num_frames_after=args.num_frames_after,
            pseudo_annos=args.pseudo_annos,
            pseudo_anno_path=args.pseudo_anno_path,
            subtitle_pkl_path=args.subtitle_pkl_path,
            episode2subset=episode2subset,
            subtitle_reference_mouthings=args.subtitle_reference_mouthings,
        )
