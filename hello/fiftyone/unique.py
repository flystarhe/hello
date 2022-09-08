import shutil
import sys
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob


def find_unique(export_dir, dataset_dir, count=1, model=None):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.ImageDirectory,
    )

    results = fob.compute_similarity(dataset, brain_key="img_sim", model=model)
    results.find_unique(count)

    unique_view = dataset.select(results.unique_ids)

    shutil.rmtree(export_dir, ignore_errors=True)

    unique_view.export(
        export_dir=(Path(export_dir) / "data").as_posix(),
        dataset_type=fo.types.ImageDirectory,
    )

    return len(unique_view)


def func(export_dir, dataset_dir, count, model):
    return find_unique(export_dir, dataset_dir, count, model)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("export_dir", type=str,
                        help="a directory")
    parser.add_argument("dataset_dir", type=str,
                        help="the dataset directory")
    parser.add_argument("-n", dest="count", type=int, default=1,
                        help="the desired number of unique examples")
    parser.add_argument("-m", dest="model", type=str, default=None,
                        help="a fiftyone.core.models.Model or the name")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
