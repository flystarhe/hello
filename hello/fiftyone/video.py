from pathlib import Path

from fiftyone.utils.video import sample_video as _sample_video


def sample_video(video_path, output_path="/tmp", fps=10, original_frame_numbers=True):
    output_patt = str(Path(output_path) / Path(video_path).stem / "%012d.png")
    return _sample_video(video_path, output_patt, fps=fps, original_frame_numbers=original_frame_numbers)


def from_tar():
    """\
    cn_courtyard_lfi_20220527_v0.01.tar

    >>> <cn_courtyard_lfi_20220527>/
    >>> ├── calib_results.txt
    >>> ├── customize.json
    >>> ├── libprocess.so
    >>> └── data
    >>>     ├── 20220527_113755.mp4
    >>>     ├── 20220527_113803.mp4
    >>>     ├── 20220527_113805.png
    >>>     └── 20220527_113814.png
    """
    return 0
