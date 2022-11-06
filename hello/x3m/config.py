from pathlib import Path

tmpl_f32 = """\
# X3M SDK: Version_20220512
# https://developer.horizon.ai/api/v1/fileData/documents/ai_toolchain_develop/horizon_ai_toolchain_user_guide/chapter_3_model_conversion.html#model-conversion
model_parameters:
  onnx_model: '{onnx_model}'
  march: 'bernoulli2'
  layer_out_dump: False
  log_level: 'debug'
  working_dir: 'model_output'
  output_model_file_prefix: '{model_prefix}'

input_parameters:
  input_name: '{input_name}'
  input_type_rt: '{input_type_rt}'
  input_layout_rt: '{input_layout_rt}'
  input_type_train: '{input_type_train}'
  input_layout_train: '{input_layout_train}'
  input_shape: ''
  input_batch: 1
  norm_type: '{norm_type}'
  mean_value: '{mean_value}'
  scale_value: '{scale_value}'

calibration_parameters:
  cal_data_dir: '{cal_data_dir}'
  calibration_type: '{calibration_type}'
  max_percentile: {max_percentile}
  per_channel: {per_channel}
  preprocess_on: False

compiler_parameters:
  compile_mode: '{compile_mode}'
  debug: {compile_debug}
  core_num: {core_num}
  optimize_level: '{optimize_level}'
"""


def todo(mode, **kwargs):
    if mode == "f32":
        config_text = tmpl_f32.format(**kwargs)
    else:
        config_text = "TBD"

    with open("config.yaml", "w") as f:
        f.write(config_text)
    print(config_text)

    print("# Build: hb_mapper makertbin --config config.yaml --model-type onnx")
    return 0


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("mode", type=str,
                        choices=["f32", "qat"])
    parser.add_argument("onnx_model", type=str,
                        help="the model file of ONNX")
    parser.add_argument("--model-prefix", type=str, default=None,
                        help="model conversion generated name prefix")

    parser.add_argument("--input-name", type=str, default="images",
                        help="node name of model input")
    parser.add_argument("--input-type-rt", type=str, default="nv12",
                        choices=["nv12", "rgb", "bgr", "gray", "featuremap", "yuv444"])
    parser.add_argument("--input-layout-rt", type=str, default="",
                        choices=["NHWC", "NCHW"])
    parser.add_argument("--input-type-train", type=str, default="rgb",
                        choices=["rgb", "bgr", "gray", "featuremap", "yuv444"])
    parser.add_argument("--input-layout-train", type=str, default="NCHW",
                        choices=["NHWC", "NCHW"])
    parser.add_argument("--norm-type", type=str, default="data_scale",
                        choices=["no_preprocess", "data_mean", "data_scale", "data_mean_and_scale"])
    parser.add_argument("--mean-value", type=str, default="",
                        help="values must be seperated by space")
    parser.add_argument("--scale-value", type=str, default="0.003921568627451",
                        help="values must be seperated by space")

    parser.add_argument("--cal-data-dir", type=str, default="./calibration_data_rgb_f32",
                        help="reference images of model quantization")
    parser.add_argument("--calibration-type", type=str, default="default",
                        choices=["kl", "max", "default", "load"])
    parser.add_argument("--max-percentile", type=str, default="1.0",
                        help="valid when `calibration_type=max`, range 0.0 - 1.0.")
    parser.add_argument("--per-channel", type=str, default="False",
                        choices=["False", "True"])

    parser.add_argument("--compile-mode", type=str, default="latency",
                        choices=["bandwidth", "lantency"])
    parser.add_argument("--compile-debug", type=str, default="False",
                        choices=["False", "True"])
    parser.add_argument("--core-num", type=str, default="1",
                        choices=["1", "2"])
    parser.add_argument("--optimize-level", type=str, default="O3",
                        choices=["O0", "O1", "O2", "O3"])

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    if kwargs["model_prefix"] is None:
        input_type = kwargs["input_type_rt"]
        model_name = Path(kwargs["onnx_model"]).stem
        kwargs["model_prefix"] = f"{model_name}_{input_type}"

    print(todo(**kwargs))

    return 0
