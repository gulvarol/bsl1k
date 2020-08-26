"""Script to download the WLASL dataset

See https://github.com/dxli94/WLASL for more details.
"""
import os
import json
import argparse
import subprocess
from pathlib import Path

import wget
from beartype import beartype


def download_youtube_video(video_identifier, output_path):
    """
        Given the youtube video_identifier, download
            using youtube-dl into output_path location.
    """
    url_base = "https://www.youtube.com/watch?v="
    command = [
        "youtube-dl",
        # '--sleep-interval', '60',
        f'"{url_base}{video_identifier}"'
        "-f",
        "mp4",
        "-o",
        f'"{output_path}"',
    ]
    command = " ".join(command)
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        # time.sleep(4)  # Otherwise HTTPError 429 (Too Many Requests)
        # https://github.com/ytdl-org/youtube-dl/issues/22382
    except subprocess.CalledProcessError as err:
        print(f"Video not downloaded: {err}.")
        pass


def compute_start_end_times(start_frame, end_frame, fps=25.0):
    start_time = (start_frame - 1) / fps
    if end_frame == -1:
        end_time = -1
    else:
        end_time = (end_frame - 1) / fps
    return start_time, end_time


@beartype
def download_youtube_clip(
    video_identifier: str,
    output_filename: str,
    start_frame,
    end_frame,
    # start_time, end_time,
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

    # Get the orig fps, sample 25 fps, convert 1-indexed frame number of sec.
    start_time, end_time = compute_start_end_times(start_frame, end_frame)
    if end_time == -1:
        t_string = ""
    else:
        t_string = f"-t {str(end_time - start_time)}"

    # Construct command to trim the videos (ffmpeg required).
    command = [
        "ffmpeg",
        "-ss",
        str(start_time),
        t_string,  # '-t', str(end_time - start_time),
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
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, "Downloaded"


@beartype
def get_youtube_identifier(url: str) -> str:
    start_ix = url.index("watch?v=") + len("watch?v=")
    end_ix = start_ix + 11
    return str(url[start_ix:end_ix])


@beartype
def download_hosted_video(video_link: str, output_path: Path):
    """
        Given the link to a video, download
            using wget into output_file location.
    """
    command = ["wget", "-O", f"{output_path}", f'"{video_link}"']
    command = " ".join(command)
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(f"Video not downloaded: {err}.")
        pass


@beartype
def download_video(inst: dict, output_path: Path):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if "youtube.com" in inst["url"]:
        video_identifier = get_youtube_identifier(inst["url"])
        download_youtube_clip(
            video_identifier, str(output_path), inst["frame_start"], inst["frame_end"]
        )
    else:
        download_hosted_video(video_link=inst["url"], output_path=output_path)


@beartype
def construct_output_path(output_dir: Path, inst: dict) -> Path:
    return output_dir / inst["split"] / f"{inst['video_id']}.mp4"


@beartype
def main(
    wlasl_dir: Path,
    wlasl_meta_data_url: str,
    wlasl_link_path: Path,
):
    info_tar = wlasl_dir / "wlasl.tar"
    output_dir = wlasl_dir / "videos_original"
    if not info_tar.exists():
        print(f"Did not find {info_tar}, downloading from {wlasl_meta_data_url}")
        info_tar.parent.mkdir(exist_ok=True, parents=True)
        wget.download(wlasl_meta_data_url, str(wlasl_dir))
        cmd = f"tar -xf {info_tar} -C {wlasl_dir}"
        os.system(cmd)

    with open(wlasl_link_path) as ipf:
        content = json.load(ipf)

    cnt = 0
    for ent in content:
        for inst in ent["instances"]:
            output_path = construct_output_path(output_dir, inst)
            if not output_path.exists():
                download_video(inst, output_path)
                if output_path.exists():
                    print(cnt, output_path)
                else:
                    print(cnt, "Not downloaded", output_path)
            cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wlasl_meta_data_url",
        type=str,
        default="https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/info/wlasl.tar",
        help="",
    )
    parser.add_argument(
        "--wlasl_dir",
        type=Path,
        default="data/wlasl",
        help="directory where the dataset will be stored",
    )
    parser.add_argument(
        "--wlasl_link_path",
        type=Path,
        default="data/wlasl/info/WLASL_v0.3.json",
        help="The latest version of the WLASL json released by the authors",
    )
    args, _ = parser.parse_known_args()
    main(**vars(args))
