import shutil
import subprocess
import sys
from pathlib import Path


def func(json_logs, out_dir, metrics, mmseg_home="/workspace", format=".png"):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=False)

    if Path(json_logs[0]).is_dir():
        json_logs = [str(f) for f in Path(json_logs[0]).glob("*/*.log.json")]
        json_logs = sorted(json_logs)

    py_script = str(Path(mmseg_home) / "tools/analyze_logs.py")

    for metric in metrics + ["mIoU", "mAcc", "aAcc"]:
        run_name_list = [Path(f).parent.name for f in json_logs]

        _arg_keys = " ".join(["--keys"] + [metric])
        _arg_title = " ".join(["--title"] + [metric])
        _arg_legend = " ".join(["--legend"] + run_name_list)
        _arg_out = " ".join(["--out"] + [str(out_dir / f"images/metric_{metric}{format}")])
        _arg_json_logs = " ".join(json_logs)

        command_line = f"python {py_script} {_arg_keys} {_arg_title} {_arg_legend} {_arg_out} {_arg_json_logs}"
        result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERR]\n  {command_line}")
    return str(out_dir)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("json_logs", type=str, nargs='+',
                        help="path of train log in json format or runs dir")
    parser.add_argument("-o", dest="out_dir", type=str,
                        help="save plotting curves to the dir")
    parser.add_argument("-m", dest="metrics", type=str, nargs='+',
                        default=["decode.loss_ce", "decode.acc_seg", "loss"],
                        help="the metric that you want to plot")
    parser.add_argument("-b", dest="mmseg_home", type=str, default="/workspace",
                        help="specify the mmseg home")
    parser.add_argument("-f", dest="format", type=str, default=".png",
                        help="image save format")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
