import subprocess
import sys
from pathlib import Path


def func(config, shape, cfg_options, mmdet_home="/workspace"):
    py_script = str(Path(mmdet_home) / "tools/analysis_tools/get_flops.py")

    _optional_args = "--size-divisor -1"

    shape = [str(v) for v in shape]
    _optional_args = " ".join([_optional_args, "--shape"] + shape)

    if cfg_options is not None and len(cfg_options) > 0:
        cfg_options = [f"'{v}'" for v in cfg_options]
        _optional_args = " ".join([_optional_args, "--cfg-options"] + cfg_options)

    command_line = f"python {py_script} {config} {_optional_args}"
    result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        print(f"[ERR]\n  {command_line}")
    else:
        print(result.stdout.decode("utf-8"))


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("config", type=str,
                        help="config file path")
    parser.add_argument("-s", dest="shape", type=int, nargs='+',
                        default=[512, 512], help="input image size")
    parser.add_argument("-e", dest="cfg_options", type=str, nargs='+',
                        help="override some settings in the used config")
    parser.add_argument("-b", dest="mmdet_home", type=str, default="/workspace",
                        help="specify the mmdet home")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
