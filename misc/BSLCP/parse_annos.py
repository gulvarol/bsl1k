"""Parse the ELAN annotations provided by the BSL Corpus dataset and store in the
same format as used by BSL-1K.

Example use:
    ipython misc/BSLCP/parse_annos.py -- --refresh --vis
"""
import re
import json
import pickle
import random
import argparse
from typing import Set, Dict, List, Tuple
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from beartype import beartype
from typeguard import typechecked
from zsvision.zs_utils import memcache
from zsvision.zs_data_structures import HashableDict

from misc.BSLCP.gloss_utils import (
    parse_glosses,
    filter_glosses,
    filter_left_right_duplicates
)


@typechecked
def parse_signers(gloss_data_filtered: Dict, raw_video_dir: Path) -> Tuple[dict, list]:
    parsed = defaultdict(lambda: {"start": [], "end": [], "video": [], "gloss": []})
    video_paths = list(raw_video_dir.glob("**/*.mov"))
    video_data = {}
    dropped = {
        "missing_vid": 0,
        "unparsable_signer": 0,
    }
    missing_videos = []
    for path in video_paths:
        video_data[path.stem] = path
    video_paths = [x.parent / f"{x.stem.replace('-comp', '')}.mov" for x in video_paths]
    for gloss, data in gloss_data_filtered.items():
        for (start, end, media) in zip(data["start"], data["end"], data["media"]):
            stems = [Path(x.replace("-comp", "").replace(".compressed", "")).stem
                     for x in media]
            assert len(stems) in {1, 2}, "Expected 1 or 2 media paths"
            if len(stems) == 2:
                focus = stems[0] if len(stems[0]) <= len(stems[1]) else stems[1]
            else:
                focus = stems[0]
            signer_regexp = r"(?P<signer>[A-Z]+[0-9]+)[a-z]+"
            matches = re.match(signer_regexp, focus)
            if matches:
                signer = matches.group("signer")
            else:
                dropped["unparsable_signer"] += 1
                continue
            if focus in video_data:
                parsed[signer]["start"].append(start)
                parsed[signer]["end"].append(end)
                parsed[signer]["gloss"].append(gloss.lower())
                parsed[signer]["video"].append(video_data[focus])
            else:
                missing_videos.append(focus)
                dropped["missing_vid"] += 1
    for key, val in dropped.items():
        print(f"annotations dropped due to {key}: {val}")
    total_found = 0
    for subdict in parsed.values():
        total_found += len(subdict["start"])
    print(f"Total annotations found: {total_found}")
    return parsed, missing_videos


@beartype
def plot_histogram(hist_counts: dict, fig_path: Path, tag: str):
    labels, values = zip(*Counter(hist_counts).most_common())
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    idx = np.arange(len(labels))
    width = 1
    ax[0].bar(idx, values, width)
    ax[0].grid()
    ax[1].bar(idx, values, width)
    ax[1].grid()
    ax[1].set_yscale("log")
    font = {"family": "serif", "weight": "normal", "size": 26}
    matplotlib.rc("font", **font)
    plt.suptitle(f"BSLCP {tag} [vocab: {len(labels)}, glosses: {sum(values)}]")
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path)


@typechecked
def dump_hist_counts(hist_counts: dict, tag: str, fig_dir: Path):
    fig_path = fig_dir / f"hists/{tag}.png"
    txt_path = fig_dir / f"{tag}.txt"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    with open(txt_path, "w") as f:
        for key, val in sorted(hist_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{key},{val}\n")
    plot_histogram(hist_counts=hist_counts, fig_path=fig_path, tag=tag)


@typechecked
def parse_annos(
        anno_dir: Path,
        fig_dir: Path,
        vocab_name: str,
        raw_video_dir: Path,
        dest_path: Path,
        canonical_vocab: Set,
        target_tiers: Tuple[str, str],
        train_val_test_ratio: List[float],
        vis: bool,
        refresh: bool,
):
    if dest_path.exists() and not refresh:
        print(f"Found existing annotations at {dest_path}, skipping...")
        return

    # fix seeds so that we can reproduce the splits
    random.seed(0)
    np.random.seed(0)

    gloss_data = parse_glosses(anno_dir=anno_dir, target_tiers=target_tiers)
    print(f"Before filtering, found {len(gloss_data)} raw glosses")
    if vis:
        hist_counts = {key: len(val["start"]) for key, val in gloss_data.items()}
        dump_hist_counts(
            tag="pre-filter",
            fig_dir=fig_dir,
            hist_counts=hist_counts,
        )

    gloss_data_filtered = filter_glosses(
        gloss_data=HashableDict(gloss_data),
        canonical_vocab=tuple(canonical_vocab),
    )
    print(f"After filtering, found {len(gloss_data_filtered)} raw glosses")

    if vis:
        hist_counts = {key: len(val["start"]) for key, val in gloss_data_filtered.items()}
        dump_hist_counts(
            tag=f"post-gloss-filter-{vocab_name}",
            fig_dir=fig_dir,
            hist_counts=hist_counts,
        )

    gloss_data_filtered_left_right = {gloss: filter_left_right_duplicates(data)
                                      for gloss, data in gloss_data_filtered.items()}
    print(f"After filtering LH, found {len(gloss_data_filtered_left_right)} raw glosses")

    if vis:
        hist_counts = {key: len(val["start"])
                       for key, val in gloss_data_filtered_left_right.items()}
        dump_hist_counts(
            tag=f"post-gloss-left-right-filter-{vocab_name}",
            fig_dir=fig_dir,
            hist_counts=hist_counts,
        )

    gloss_by_signer, missing_videos = parse_signers(
        raw_video_dir=raw_video_dir,
        gloss_data_filtered=gloss_data_filtered_left_right,
    )
    missing_video_path = fig_dir / "missing-videos.txt"
    print(f"Writing missing video list to {missing_video_path}")
    with open(missing_video_path, "w") as f:
        for missing_video in sorted(set(missing_videos)):
            f.write(f"{missing_video}\n")

    if vis:
        hist_counts = defaultdict(lambda: 0)
        for subdict in gloss_by_signer.values():
            for gloss in subdict["gloss"]:
                hist_counts[gloss] += 1
        dump_hist_counts(
            tag=f"post-video-existence-filter-{vocab_name}",
            fig_dir=fig_dir,
            hist_counts=hist_counts,
        )

    signers = list(gloss_by_signer.keys())
    np.random.shuffle(signers)
    subset_data = {
        "train": defaultdict(lambda: {"start": [], "end": [], "video": []}),
        "val": defaultdict(lambda: {"start": [], "end": [], "video": []}),
        "test": defaultdict(lambda: {"start": [], "end": [], "video": []}),
    }
    num_train = int(np.round(train_val_test_ratio[0] * len(signers)))
    num_val = int(np.round(train_val_test_ratio[1] * len(signers)))
    num_test = len(signers) - (num_train + num_val)
    print(f"Assigning {num_train} train, {num_val} val, {num_test} test")
    signer_ids = {
        "train": signers[:num_train],
        "val": signers[num_train:num_train + num_val],
        "test": signers[num_train + num_val:],
    }
    for key, vals in signer_ids.items():
        print(f"{key}: {len(vals)}")

    for subset, signers in signer_ids.items():
        for signer in signers:
            subdict = gloss_by_signer[signer]
            for (start, end, video, gloss) in zip(subdict["start"], subdict["end"],
                                                  subdict["video"], subdict["gloss"]):
                subset_data[subset][gloss]["start"].append(start)
                subset_data[subset][gloss]["end"].append(end)
                subset_data[subset][gloss]["video"].append(str(video))

    print(f"Writing full annotations to {dest_path}")
    with open(dest_path, "w") as f:
        json.dump(dict(subset_data), f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fig_dir", type=Path, default="misc/BSLCP/figs")
    parser.add_argument("--config", type=Path, default="misc/BSLCP/data_paths.json")
    parser.add_argument(
        "--vocab_name",
        default="bsl1k_vocab",
        choices=["bsl1k_vocab", "BSLCP_all_glosses", "signdict_signbank"],
    )
    args = parser.parse_args()

    config = memcache(args.config)

    dest_path = Path(config[args.vocab_name]["anno_path"])
    vocab_path = config[args.vocab_name]["vocab_path"]
    if vocab_path:
        with open(vocab_path, "rb") as f:
            canonical_vocab = set(pickle.load(f)["words"])
    else:
        # We use an empty vocabulary to denote that no filtering should be performed
        canonical_vocab = set()
    fig_dir = args.fig_dir / args.vocab_name

    parse_annos(
        anno_dir=Path(config["raw_anno_dir"]),
        target_tiers=tuple(config["target_tiers"]),
        train_val_test_ratio=config["train_val_test_ratio"],
        raw_video_dir=Path(config["raw_video_dir"]),
        vocab_name=args.vocab_name,
        fig_dir=fig_dir,
        dest_path=dest_path,
        canonical_vocab=canonical_vocab,
        refresh=args.refresh,
        vis=args.vis,
    )


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
