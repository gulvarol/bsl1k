"""Script to extract clips from the BSL-1K annotated split.

Example usage:
    ipython misc/bbcsl_annotated/extract_clips.py
"""
import argparse
import os
import pickle as pkl
import subprocess
from collections import defaultdict


def extract_clip(source_file, output_filename, start_time, end_time):
    """Extract a clip from a video.

    arguments:
    ---------
    source_file: str
        Path to untrimmed video
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(output_filename, str), "output_filename must be string"
    status = False
    # Construct command to trim the videos (ffmpeg required).
    command = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-t",
        str(end_time - start_time),
        "-i",
        "'%s'" % source_file,
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-threads",
        "1",
        "-loglevel",
        "panic",
        '"%s"' % output_filename,
    ]
    command = " ".join(command)
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status


def take_interval_from_peak(time_in_sec, fps=25, num_frames=64):
    padding = num_frames / fps
    start_time = time_in_sec - padding
    end_time = time_in_sec
    return start_time, end_time


def construct_video_filename(
    output_dir, set_name, word, name, start_time, end_time, trim_format
):
    output_filename = "%s_%s_%s.mp4" % (
        name,
        trim_format % start_time,
        trim_format % end_time,
    )
    return os.path.join(output_dir, set_name, word, output_filename)


def construct_original_filename(name, use_fixed_videos):
    # videos_parent_orig = '/scratch/shared/beegfs/albanie/exp/kws-align/bbcsl/videos-mp4'
    videos_parent_orig = (
        "/scratch/shared/beegfs/albanie/shared-datasets/bbcsl/videos-mp4/"
    )
    parent_folder, base_folder = name.split("--")
    if use_fixed_videos:
        fname = "signhd-dense-fast-audio.mp4"
    else:
        fname = "signhd.mp4"
    return os.path.join(videos_parent_orig, parent_folder, base_folder, fname)


def mkdir_p(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(output_dir, prob_thres, trim_format, anno_pkl_path, use_fixed_videos, refresh):
    video_folder = "annotated-videos"
    if use_fixed_videos:
        video_folder += "-fixed"
    output_dir_videos = os.path.join(output_dir, video_folder)
    # pkl_file = os.path.join(output_dir, 'info', 'mouthings_2020.01.27.pkl')
    data = pkl.load(open(anno_pkl_path, "rb"))
    count = 0
    for worddict in data["test"].values():
        count += len(worddict["probs"])
    print(f"Count: {count}")

    cnts = {}
    output_filenames = defaultdict(list)
    failed = []
    for s in data.keys():
        cnts[s] = []
        for word in data[s].keys():
            cnts[s].append(0)
            mkdir_p(os.path.join(output_dir_videos, s, word))
            N = len(data[s][word]["names"])
            for i in range(N):
                if data[s][word]["probs"][i] > prob_thres:
                    cnts[s][-1] += 1
                    start_time, end_time = take_interval_from_peak(
                        data[s][word]["times"][i]
                    )
                    output_filename = construct_video_filename(
                        output_dir_videos,
                        s,
                        word,
                        data[s][word]["names"][i],
                        start_time,
                        end_time,
                        trim_format,
                    )
                    output_filenames[output_filename].append((start_time, end_time))
                    if not os.path.exists(output_filename) or refresh:
                        source_file = construct_original_filename(
                            data[s][word]["names"][i], use_fixed_videos=use_fixed_videos
                        )
                        status = extract_clip(
                            source_file, output_filename, start_time, end_time
                        )
                        print(output_filename, source_file)
                        print(status)
                        if not status:
                            failed.append(source_file)
    fnames = sorted(output_filenames, key=lambda x: len(x[1]))
    for fname in fnames[:10]:
        print(f"{fname}: {output_filenames[fname]}")


if __name__ == "__main__":
    description = "Script to extract clips from the BSL-1K annotated split."
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/bsl1k_annotated",
        help="Output directory where videos will be saved.",
    )
    p.add_argument(
        "--prob_thres",
        type=float,
        default=0.5,
        help="Threshold for the mouthing probability.",
    )
    p.add_argument(
        "--use_fixed_videos",
        type=int,
        default=1,
        help="whether to use the corrected videos",
    )
    p.add_argument(
        "--refresh", action="store_true", help="whether to refresh the videos"
    )
    p.add_argument(
        "--anno_pkl_path",
        default="data/bsl1k_annotated/info/annotations_in_mouthing_format_2020.03.04.pkl",
        help="Location of the pkl file containing the human verified annotations.",
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
    main(**vars(p.parse_args()))
