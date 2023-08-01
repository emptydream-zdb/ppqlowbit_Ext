# PPQ TRT_Lowbit_Exp

对ppq的低比特量化扩展，未更改ppq源码，导入即可使用

## requirements

```text
ppq
torch
```
## 结构以及主要api

```text
ppq_lowbit
├── __init__.py  ———— 导入包，在这里将自定义量化器注册到 TargetPlatform.TRT_INT8，导出器注册到ONNXRUNTIME
├── api.py  ———— 定义自己的量化接口
├── onnxruntime_lowbit_exporter.py  ———— 导出器实现
└── trt_lowbit_quantizer.py  ———— 量化器实现
```
api.py中:

- quantize_onnx_model_lowbit()  量化原始onnx模型，可任意比特

- quantize_native_model_lowbit()  量化ppl native 图格式模型，可任意比特

- quantize_onnx_model_lowbit_cus() 量化原始onnx模型，可任意比特，且激活和权重可不同比特

- quantize_native_model_lowbit_cus()量化ppl native 图格式模型，可任意比特，且激活和权重可不同比特

**具体传入参数可阅读api文件中的源码**

## tips
- onnx 原生不支持低于int8的数据类型，所以导出的量化图中显示全为int8/uint8类型，实际量化是没有问题的
- 复杂的网络比如yolo，需要手动的去裁剪修改网络的一些层和输出，否则量化出错，这个问题是量化操作的局限
- 该扩展很好的融入了ppq, 除了使用自定义的 api 外，其余的所有操作完全不用更改ppq源码
- 由于对ppq不是完全了解，测试不全，可能在运行过程中存在各种问题，若发现问题请及时联系
- **新版加入了激活和权重不同位宽量化的支持**

## Example

- 导出图使用ppq原生接口即可，选择ONNXRUNTIME平台。

```python
from ppq_lowbit import *   #导入包
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cuda'  # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.TRT_INT8  # 选择TRT_INT8 平台
ONNX_PATH = 'models/model.onnx'


def load_calibration_dataset() -> Iterable: # 初始化随机数 Dataset
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


quant_setting = QuantizationSettingFactory.trt_setting()

# Load training data for creating a calibration dataloader.
calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(  # 初始化Dataloader
    dataset=calibration_dataset,
    batch_size=BATCHSIZE, shuffle=True)

# quantize your model.
quantized = quantize_onnx_model_lowbit(  # 调用api完成量化
    onnx_import_file=ONNX_PATH, bit_width=4, symmetrical=True, per_channel=True,
    calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE,
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    device=DEVICE, verbose=0)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph. 使用ppq原生导出器导出到ONNXRUNTIME平台即可，一定要是ONNXRUNTIME平台
export_ppq_graph(graph=quantized, platform=TargetPlatform.ONNXRUNTIME, 
                 graph_save_to='Output/quantized(onnx).onnx',
                 config_save_to='Output/quantized(onnx).json')
```