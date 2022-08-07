from pathlib import Path

import cv2 as cv

suffix_set = set(".avi,.mp4".split(","))


def video_info(fpath):
    cap = cv.VideoCapture(fpath)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    cap.release()
    return fps, width, height, count


def find_videos(input_path, end_tag="_tof"):
    video_paths = []
    for fpath in sorted(Path(input_path).glob("**/*")):
        if fpath.suffix not in suffix_set:
            continue
        video_paths.append(fpath)

    links = {}
    main_files = []
    len_tag = len(end_tag)
    for fpath in video_paths:
        fstem = fpath.stem
        if fstem.endswith(end_tag):
            links[fstem[:-len_tag]] = fpath
        else:
            main_files.append((fstem, fpath))

    video_pairs = []
    for fstem, fpath in main_files:
        video_pairs.append((fpath, links.get(fstem)))
    return video_pairs


def align_pairs(video_pairs, output_dir):
    return video_pairs


def display_pairs(frame1, frame2=None, height=800):
    return None


if __name__ == "__main__":
    res = find_videos("/workspace/tmp/rgb_tof_2022_08_04_pm", "_g")
    print(len(res), "\n", res)
