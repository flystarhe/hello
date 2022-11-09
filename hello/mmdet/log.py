import shutil
import subprocess
import sys
from pathlib import Path


def func(json_logs, out_dir, metrics=["loss", "loss_cls", "loss_bbox"], mmdet_home="/workspace", format=".png"):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=False)

    if Path(json_logs[0]).is_dir():
        json_logs = [str(f) for f in Path(json_logs[0]).glob("*/*.log.json")]
        json_logs = sorted(json_logs)

    py_script = str(Path(mmdet_home) / "tools/analysis_tools/analyze_logs.py")

    for f in json_logs:
        command_line = f"python {py_script} cal_train_time {f}"
        result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERR]\n  {command_line}")

        run_name = Path(f).parent.name

        _arg_keys = " ".join(["--keys"] + metrics)
        _arg_title = " ".join(["--title"] + [run_name])
        _arg_legend = " ".join(["--legend"] + metrics)
        _arg_out = " ".join(["--out"] + [str(out_dir / f"images/run_{run_name}{format}")])
        _arg_json_logs = " ".join([f])

        command_line = f"python {py_script} plot_curve {_arg_keys} {_arg_title} {_arg_legend} {_arg_out} {_arg_json_logs}"
        result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERR]\n  {command_line}")

    for metric in metrics + ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75"]:
        run_name_list = [Path(f).parent.name for f in json_logs]

        _arg_keys = " ".join(["--keys"] + [metric])
        _arg_title = " ".join(["--title"] + [metric])
        _arg_legend = " ".join(["--legend"] + run_name_list)
        _arg_out = " ".join(["--out"] + [str(out_dir / f"images/metric_{metric}{format}")])
        _arg_json_logs = " ".join(json_logs)

        command_line = f"python {py_script} plot_curve {_arg_keys} {_arg_title} {_arg_legend} {_arg_out} {_arg_json_logs}"
        result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERR]\n  {command_line}")


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("json_logs", type=str, nargs='+',
                        help="path of train log in json format or runs dir")
    parser.add_argument("-o", dest="out_dir", type=str,
                        help="save plotting curves to the dir")
    parser.add_argument("-m", dest="metrics", type=str, nargs='+',
                        default=["loss", "loss_cls", "loss_bbox"],
                        help="the metric that you want to plot")
    parser.add_argument("-b", dest="mmdet_home", type=str, default="/workspace",
                        help="specify the mmdet home")
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
