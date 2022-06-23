# ONNX

## Export ONNX
```python
import torch
from hello.onnx.export import Net, export_onnx

model = Net()
x = torch.randn(1, 3, 224, 224)
export_onnx(model, x, "test.onnx", device="cuda:0")
```
