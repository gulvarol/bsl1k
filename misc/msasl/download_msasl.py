import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import wget
import pandas as pd
from joblib import Parallel, delayed
from beartype import beartype


def construct_video_filename(output_dir, row, trim_format="%06d"):
    """Given a dataset row, this function constructs the 
       output filename for a given video.
    """
    output_filename = "%s_%s_%s.mp4" % (
        row["video-id"],
        trim_format % row["start-time"],
        trim_format % row["end-time"],
    )
    return os.path.join(output_dir, output_filename)


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir="/tmp/download",
    num_attempts=5,
    url_base="https://www.youtube.com/watch?v=",
):
    """Download a video from youtube if exists and is not blocked.
    
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video 
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), "video_identifier must be string"
    assert isinstance(output_filename, str), "output_filename must be string"
    assert len(video_identifier) == 11, "video_identifier must have length 11"

    status = False
    # Construct command line for getting the direct video link.
    command = [
        "youtube-dl",
        "--quiet",
        "--no-warnings",
        "-f",
        "mp4",
        "--get-url",
        '"%s"' % (url_base + video_identifier),
    ]
    command = " ".join(command)
    attempts = 0
    while True:
        try:
            direct_download_url = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
            direct_download_url = direct_download_url.strip().decode("utf-8")
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Construct command to trim the videos (ffmpeg required).
    command = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-t",
        str(end_time - start_time),
        "-i",
        "'%s'" % direct_download_url,
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
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, "Downloaded"


def download_clip_wrapper(row, trim_format, tmp_dir, output_dir):
    """Wrapper for parallel processing purposes."""
    output_filename = construct_video_filename(output_dir, row, trim_format)
    Path(output_filename).parent.mkdir(exist_ok=True, parents=True)
    clip_id = os.path.basename(output_filename).split(".mp4")[0]
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, "Exists"])
        return status

    downloaded, log = download_clip(
        row["video-id"],
        output_filename,
        row["start-time"],
        row["end-time"],
        tmp_dir=tmp_dir,
    )
    print(log)
    status = tuple([clip_id, downloaded, log])
    return status


@beartype
def parse_annotations(input_json: Path) -> pd.DataFrame:
    """Returns a parsed DataFrame.

    arguments:
    ---------
    input_json: path to JSON file containing the following columns:
               'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_json(input_json)
    youtube_ids = []
    for i in range(len(df)):
        start_ix = df["url"][i].index("watch?v=") + len("watch?v=")
        end_ix = start_ix + 11
        youtube_ids.append(df["url"][i][start_ix:end_ix])

    df.insert(0, "video-id", youtube_ids)
    df.rename(columns={"start_time": "start-time", "end_time": "end-time"}, inplace=True)
    return df


@beartype
def main(
    input_jsons: list,
    msasl_dir: Path,
    msasl_meta_data_url: str,
    output_dir: Path,
    trim_format: str,
    num_jobs: int,
    tmp_dir: Path,
):
    info_tar = msasl_dir / "msasl.tar"
    output_dir = msasl_dir / "videos_original"
    if not info_tar.exists():
        print(f"Did not find {info_tar}, downloading from {msasl_meta_data_url}")
        info_tar.parent.mkdir(exist_ok=True, parents=True)
        wget.download(msasl_meta_data_url, str(msasl_dir))
        cmd = f"tar -xf {info_tar} -C {msasl_dir}"
        os.system(cmd)

    # Reading and parsing annotations.
    for input_json in input_jsons:
        dataset = parse_annotations(input_json)
        # Download all clips.
        if num_jobs == 1:
            status_lst = []
            for i, row in dataset.iterrows():
                status_lst.append(
                    download_clip_wrapper(row, trim_format, tmp_dir, output_dir)
                )
        else:
            status_lst = Parallel(n_jobs=num_jobs)(
                delayed(download_clip_wrapper)(row, trim_format, tmp_dir, output_dir)
                for i, row in dataset.iterrows()
            )
        # Clean tmp dir.
        shutil.rmtree(tmp_dir)

        # Save download report.
        with open("download_report.json", "w") as fobj:
            fobj.write(json.dumps(status_lst))


if __name__ == "__main__":
    description = "Helper script for downloading and trimming kinetics videos."
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--msasl_meta_data_url",
        type=str,
        default="https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/info/msasl.tar",
        help="",
    )
    p.add_argument(
        "--msasl_dir",
        type=Path,
        default="data/msasl",
        help="directory where the dataset will be stored",
    )
    p.add_argument(
        "--input_jsons",
        nargs="+",
        type=Path,
        default=[
            Path("data/msasl/info/MSASL_train.json"),
            Path("data/msasl/info/MSASL_val.json"),
            Path("data/msasl/info/MSASL_test.json"),
        ],
        help=(
            "JSON files containing the following format: "
            "YouTube Identifier,Start time,End time,Class label"
        ),
    )
    p.add_argument(
        "--output_dir",
        default="data/msasl/videos_original",
        type=Path,
        help="Output directory where videos will be saved."
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
    p.add_argument("-n", "--num-jobs", type=int, default=24)
    p.add_argument("-t", "--tmp-dir", type=Path, default="/tmp/download")
    main(**vars(p.parse_args()))
