from .trt_lowbit_quantizer import TrtLowBitQuantizer, TrtLowBitQuantizer_cus
from .onnxruntime_lowbit_exporter import ONNXRUNTIMLowBitExporter
from ppq.lib.extension import (register_network_quantizer,
                               register_network_exporter)
from ppq.core import TargetPlatform
from .api import (quantize_onnx_model_lowbit,
                  quantize_onnx_model_lowbit_cus,
                  quantize_native_model_lowbit,
                  quantize_native_model_lowbit_cus)

register_network_quantizer(TrtLowBitQuantizer, TargetPlatform.TRT_INT8)
register_network_exporter(ONNXRUNTIMLowBitExporter, TargetPlatform.ONNXRUNTIME)
