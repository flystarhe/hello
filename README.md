# Hello

## Publish Package
Package is at https://pypi.org/project/hello2/

```sh
# https://github.com/pypa/flit
flit publish
```

## requirements.txt
```sh
# requirements.txt
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python Pillow scikit-image scikit-learn simplejson onnx

# hello
pip install -U hello2
pip install -U hello2 -i https://pypi.org/simple
pip install -U hello2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U 'git+https://github.com/flystarhe/hello'

# FFmpeg
apt install -y ffmpeg
conda install -c pytorch ffmpeg
conda install -c conda-forge ffmpeg

# OpenCV
pip uninstall -y opencv-python-headless
pip install opencv-python

# fiftyone
pip install fiftyone==0.16.5
pip install fiftyone[desktop]==0.16.5

# pyomniunwarp
pip install -U pyomniunwarp>=0.2.4

# onnxruntime
pip install onnx onnx-simplifier onnxruntime  # CPU
pip install onnx onnx-simplifier onnxruntime-gpu  # GPU

# PyTorch 1.10.2
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# PyTorch 1.12.1
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
