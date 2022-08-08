import shutil
import sys
from pathlib import Path

import cv2 as cv
from moviepy import editor as mpe

suffix_set = set(".avi,.mp4".split(","))


def video_info(fpath):
    cap = cv.VideoCapture(fpath)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    cap.release()
    return fps, width, height, count


def find_videos(input_dir, end_tag="_tof"):
    video_paths = []
    for fpath in sorted(Path(input_dir).glob("**/*")):
        if fpath.suffix not in suffix_set:
            continue
        video_paths.append(fpath)

    links = {}
    main_files = []
    len_tag = len(end_tag)
    for fpath in video_paths:
        fstem = fpath.stem
        if fstem.endswith(end_tag):
            links[fstem[:-len_tag]] = fpath.as_posix()
        else:
            main_files.append((fstem, fpath.as_posix()))

    video_pairs = []
    for fstem, fpath in main_files:
        video_pairs.append((fpath, links.get(fstem)))
    return video_pairs


def align_pairs(video_pairs, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    for file1, file2 in video_pairs:
        _align_pairs(file1, file2, output_dir)

    return output_dir


def _align_pairs(infile1, infile2, outdir):
    info_1 = video_info(infile1)

    info_2 = info_1
    if infile2 is not None:
        info_2 = video_info(infile2)

    s1 = int(info_1[-1] / info_1[0])
    s2 = int(info_2[-1] / info_2[0])
    s = min(s1, s2)

    _clip_video(infile1, 0, s, outdir=outdir)
    if infile2 is not None:
        _clip_video(infile2, 0, s, outdir=outdir)


def _clip_video(infile, t_start, t_end, outdir):
    clip = mpe.VideoFileClip(infile, audio=False).subclip(t_start, t_end)
    outfile = str(Path(outdir) / Path(infile).name)
    clip.write_videofile(outfile)


def func(input_dir, output_dir, end_tag):
    video_pairs = find_videos(input_dir, end_tag=end_tag)
    res = align_pairs(video_pairs, output_dir)
    return res


def parse_args(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Align Video pairs")

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("output_dir", type=str,
                        help="output dir")
    parser.add_argument("--end-tag", type=str, default="_tof",
                        help="the pair video file")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
