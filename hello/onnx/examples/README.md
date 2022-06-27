# ONNX

## Export ONNX
```python
import torch
from hello.onnx.export import Net, export_onnx

model = Net()
x = torch.randn(1, 3, 16, 16)
export_onnx(model, x, "test_16x16.onnx", device="cpu", simplify=False)
```
