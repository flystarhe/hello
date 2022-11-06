import shutil
import sys
from pathlib import Path

import cv2 as cv

suffix_set = set(".avi,.mp4,.MOV,.mkv".split(","))


def video_info(fpath):
    cap = cv.VideoCapture(fpath)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return fps, count, width, height


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


def align_pairs(infile1, infile2, outdir):
    info_1 = video_info(infile1)

    info_2 = info_1
    if infile2 is not None:
        info_2 = video_info(infile2)

    s1 = info_1[1] // info_1[0]
    s2 = info_2[1] // info_2[0]
    s = min(s1, s2)

    if s < 1:
        return None

    clip_video_opencv(outdir, infile1, 0, s)
    if infile2 is not None:
        clip_video_opencv(outdir, infile2, 0, s)


def clip_video_opencv(outdir, infile, t_start, t_end):
    outfile = Path(outdir) / f"data/{Path(infile).stem}.mp4"
    print(f"\n[CLIP]\n in: {infile}\nout: {outfile}")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    cap = cv.VideoCapture(infile)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter(str(outfile), fourcc, fps, (width, height))

    a, b = t_start * fps, t_end * fps

    curr_pos = 0
    while curr_pos < count:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if a <= curr_pos < b:
            out.write(frame)

        curr_pos += 1

    cap.release()
    out.release()


def func(input_dir, output_dir, end_tag):
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=False)

    video_pairs = find_videos(input_dir, end_tag=end_tag)

    for file1, file2 in video_pairs:
        align_pairs(file1, file2, output_dir)

    return f"\n[OUTDIR]\n{output_dir}"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=str,
                        help="videos dir")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="output dir")
    parser.add_argument("-e", "--end-tag", type=str, default="_tof",
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
