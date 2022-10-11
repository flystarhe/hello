# Hello
A collection of useful tools!

## Publish
[hello2 · PyPI](https://pypi.org/project/hello2/)
```sh
# https://github.com/pypa/flit
flit publish
```

## Environment
```sh
conda info -e
conda create -y -n myenv python=3.9
conda activate myenv

# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install notebook

conda deactivate
conda remove -y -n myenv --all
conda info -e
```

## Installation
```sh
# requirements.txt
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python Pillow scikit-image scikit-learn simplejson onnx

# FFmpeg
apt install -y ffmpeg
conda install -c pytorch ffmpeg
conda install -c conda-forge ffmpeg

# OpenCV
pip uninstall -y opencv-python-headless
pip install opencv-python --ignore-installed

# fiftyone
pip install fiftyone>=0.17.2
pip install fiftyone[desktop]>=0.17.2
## $ conda list | grep voxel
## $ conda list | grep fiftyone

# pyomniunwarp
pip install -U pyomniunwarp>=0.2.4

# onnxruntime (optional)
pip install onnx onnx-simplifier onnxruntime  # CPU
pip install onnx onnx-simplifier onnxruntime-gpu  # GPU

# PyTorch 1.10.2 (optional)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# PyTorch 1.12.1 (optional)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# hello
pip install -U hello2
pip install -U hello2 -i https://pypi.org/simple
pip install -U hello2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U 'git+https://github.com/flystarhe/hello'
```

## Usage

### hello-data
- `hello-data coco2yolo -h`
    - COCO format to YOLOv5

### hello-fiftyone
- For examples: [hello/fiftyone/examples/](https://github.com/flystarhe/hello/blob/main/hello/fiftyone/examples)

### hello-onnx
- For examples: [hello/onnx/examples/](https://github.com/flystarhe/hello/tree/main/hello/onnx/examples)

### hello-video
- `hello-video clip -h`
- `hello-video frames -h`

### hello-x3m
- `hello-x3m preprocess -h`
    - 为X3M量化步骤生成校准数据
- `hello-x3m config -h`
    - 为X3M编译步骤生成配置文件
