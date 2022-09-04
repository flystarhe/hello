# Hello
A collection of useful tools!

```sh
pip install -U hello2
pip install -U hello2 -i https://pypi.org/simple
pip install -U hello2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U 'git+https://github.com/flystarhe/hello'
```

## hello.data
- `python -m hello.data coco2yolo -h`
    - COCO format to YOLOv5

## hello.fiftyone
- For examples: [hello/fiftyone/examples/](https://github.com/flystarhe/hello/blob/main/hello/fiftyone/examples)

## hello.onnx
- For examples: [hello/onnx/examples/](https://github.com/flystarhe/hello/tree/main/hello/onnx/examples)

## hello.video
- `python -m hello.video.clip -h`
- `python -m hello.video.frames -h`

## hello.x3m
- `python -m hello.x3m preprocess -h`
    - 为X3M量化步骤生成校准数据
- `python -m hello.x3m config -h`
    - 为X3M编译步骤生成配置文件
