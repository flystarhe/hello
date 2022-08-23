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

# https://ffmpeg.org/ffmpeg.html
#     1. apt install -y ffmpeg
#     2. conda install -c pytorch ffmpeg
#     3. conda install -c conda-forge ffmpeg

pip install opencv-python Pillow scikit-image scikit-learn pycocotools simplejson onnx onnx-simplifier onnxruntime-gpu

pip install -U hello2
pip install -U fiftyone>=0.16.5
pip install -U pyomniunwarp>=0.2.4

# PyTorch 1.10.2
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# PyTorch 1.12.1
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
