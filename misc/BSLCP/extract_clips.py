"""Script to extract clips from the BSLCP dataset.

Example usage:
    ipython misc/BSLCP/extract_clips.py
"""
import argparse
import multiprocessing as mp
from typing import Tuple
from pathlib import Path
from collections import defaultdict

import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from beartype import beartype
from typeguard import typechecked
from zsvision.zs_utils import memcache
from zsvision.zs_multiproc import starmap_with_kwargs

from misc.bsl1k.extract_clips import extract_clip, construct_video_filename


@typechecked
def time2tuple(time_secs: float) -> Tuple[float, int]:
    ms = int(np.round(1000 * (time_secs - int(time_secs))))
    return (time_secs, ms)


@beartype
def main(
        video_dir: Path,
        trim_format: str,
        pad_clip: float,
        limit: int,
        processes: int,
        json_anno_path: Path,
        anno_name: str,
        force_resize: int,
        refresh: bool,
        vis: bool,
):
    print(f"Processing {anno_name} annotations")
    data = memcache(json_anno_path)

    output_filenames = defaultdict(list)
    kwarg_list = []
    outs = set()
    count = 0
    for s in tqdm.tqdm(data.keys()):
        for word in tqdm.tqdm(data[s].keys()):
            N = len(data[s][word]["start"])
            for i in range(N):
                start_time = data[s][word]["start"][i] - pad_clip
                end_time = data[s][word]["end"][i] + pad_clip
                output_filename = construct_video_filename(
                    output_dir=video_dir,
                    set_name=s,
                    word=word,
                    name=Path(data[s][word]["video"][i]).stem,
                    start_time=time2tuple(start_time),
                    end_time=time2tuple(end_time),
                    trim_format=trim_format,
                )
                output_filenames[output_filename].append((start_time, end_time))
                source_file = Path(data[s][word]["video"][i])
                assert source_file.exists(), f"Expected source file at {source_file}"
                kwargs = {
                    "refresh": refresh,
                    "start_time": start_time,
                    "end_time": end_time,
                    "output_filename": output_filename,
                    "source_file": source_file,
                    "force_resize": force_resize,
                }
                outs.add(output_filename)
                kwarg_list.append(kwargs)
                count += 1

    if vis:
        durations = np.array([x["end_time"] - x["start_time"] for x in kwarg_list])
        step = 0.1
        bins = np.arange(0, np.ceil(durations.max()), step=step)
        values, _ = np.histogram(durations, bins=bins)
        plt.figure(figsize=(20, 10))
        x_ticks = bins[:-1] + (step / 2)
        plt.bar(x_ticks, values, width=step)
        font = {"family": "serif", "weight": "normal", "size": 26}
        matplotlib.rc("font", **font)
        plt.suptitle(f"BSLCP sign durations")
        plt.savefig("zz-bslcp-durations.png")

    if limit:
        kwarg_list = kwarg_list[:limit]
    func = extract_clip
    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)
    else:
        for kwargs in tqdm.tqdm(kwarg_list):
            func(**kwargs)
    print(f"Expected to produce: {len(kwarg_list)} outputs")


if __name__ == "__main__":
    matplotlib.use("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--config", type=Path, default="misc/BSLCP/data_paths.json")
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--anno_name", default="BSLCP_all_glosses",
                   choices=["bsl1k_vocab", "BSLCP_all_glosses", "signdict_signbank"])
    args = p.parse_args()
    p_kwargs = vars(args)
    config = memcache(p_kwargs.pop("config"))
    p_kwargs.update({
        "force_resize": config["force_resize"],
        "json_anno_path": Path(config[args.anno_name]["anno_path"]),
        "trim_format": config["trim_format"],
        "video_dir": Path(config["data_dir"]) / config[args.anno_name]["video_dir"],
        "pad_clip": config["pad_clip"],
    })
    main(**p_kwargs)
