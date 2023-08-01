from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, Operation

from ppq.quantization.quantizer.base import BaseQuantizer


class TrtLowBitQuantizer(BaseQuantizer):
    """低比特量化器定义"""
    def __init__(
        self, graph: BaseGraph,per_channel: bool = True,
        symmetrical: bool = False, num_of_bits = 8
    )-> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self.num_of_bits = num_of_bits
        self.per_channel = per_channel
        self.symmetric = symmetrical
        if symmetrical:
            self._quant_min = -pow(2, num_of_bits - 1)
            self._quant_max = pow(2, num_of_bits - 1) - 1
            self.policy    = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR +
                QuantizationProperty.SYMMETRICAL)
        else:
            self._quant_min = 0
            self._quant_max = pow(2, num_of_bits) - 1
            self.policy    = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR +
                QuantizationProperty.ASYMMETRICAL)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

        OQC = self.create_default_quant_config(
            policy=self.policy, op=operation,
            rounding=self.rounding_policy,
            num_of_bits=self.num_of_bits,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul',
                              'PPQBiasFusedMatMul'}:
            # base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                if operation.inputs[1].is_parameter:
                    conv_weight_config = OQC.input_quantization_config[1]
                    # for both SYMMETRICAL and ASYMMETRICAL quantization,
                    # weight should always be quantized symmetrically.
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        ( QuantizationProperty.PER_CHANNEL if
                          self.per_channel else QuantizationProperty.PER_TENSOR)
                    )
                    conv_weight_config.quant_min = -pow(2, self.num_of_bits - 1)
                    conv_weight_config.quant_max = pow(2, self.num_of_bits - 1) - 1
                    if self.per_channel:
                        conv_weight_config.channel_axis = (
                            1 if operation.type == 'ConvTranspose' else 0)
                    conv_weight_config.observer_algorithm = 'minmax'

            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = OQC.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        (QuantizationProperty.PER_CHANNEL if
                         self.per_channel else QuantizationProperty.PER_TENSOR)
                    )
                    gemm_weight_config.quant_min = -pow(2, self.num_of_bits - 1)
                    gemm_weight_config.quant_max = pow(2, self.num_of_bits - 1) - 1
                    if self.per_channel:
                        gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = 'minmax'

            if operation.num_of_input > 2:
                bias_config = OQC.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type == 'LayerNormalization':
            # Layernorm - gamma and beta need to be FP32
            for TQC in OQC.input_quantization_config[1:]:
                TQC.state = QuantizationStates.FP32

        if operation.type == 'Attention':
            # Attention - Only input and weight need to be quantized.
            for TQC in OQC.input_quantization_config[2:]:
                TQC.state = QuantizationStates.FP32

        return OQC

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.TRT_INT8

    @property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool', 'Softmax',
            'Mul', 'Add', 'Max', 'Sub', 'Div', 'Reshape',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Interp',
            'ReduceMean', 'Transpose', 'Slice', 'Flatten',
            'HardSwish', 'HardSigmoid', 'MatMul',
            'Attention', 'LayerNormalization', 'Gelu',
            'PPQBiasFusedMatMul'
        }

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip', 'Swish', 'Clip', 'SoftPlus', 'Sigmoid', 'Gelu'}


class TrtLowBitQuantizer_cus(BaseQuantizer):
    """可定制权重和激活不同位宽量化, 目前只支持激活位宽比权重位宽大"""

    def __init__(
            self, graph: BaseGraph, per_channel: bool = True,
            symmetrical: bool = False, bit_w = 8,bit_in = 8
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self.bit_w = bit_w
        self.bit_in = bit_in
        self.per_channel = per_channel
        self.symmetric = symmetrical
        if symmetrical:
            self._quant_min = -pow(2, bit_in - 1)
            self._quant_max = pow(2, bit_in - 1) - 1
            self.policy = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR +
                QuantizationProperty.SYMMETRICAL)
        else:
            self._quant_min = 0
            self._quant_max = pow(2, bit_in) - 1
            self.policy = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR +
                QuantizationProperty.ASYMMETRICAL)

    def init_quantize_config(self,
                             operation: Operation) -> OperationQuantizationConfig:
        OQC = self.create_default_quant_config(
            policy=self.policy, op=operation,
            rounding=self.rounding_policy,
            num_of_bits=self.bit_in,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul',
                              'PPQBiasFusedMatMul'}:
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                if operation.inputs[1].is_parameter:
                    conv_weight_config = OQC.input_quantization_config[1]
                    # for both SYMMETRICAL and ASYMMETRICAL quantization,
                    # weight should always be quantized symmetrically.
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        (QuantizationProperty.PER_CHANNEL if
                         self.per_channel else QuantizationProperty.PER_TENSOR)
                    )
                    conv_weight_config.num_of_bits = self.bit_w
                    conv_weight_config.quant_min = -pow(2, self.bit_w - 1)
                    conv_weight_config.quant_max = pow(2,
                                                       self.bit_w - 1) - 1
                    if self.per_channel:
                        conv_weight_config.channel_axis = (
                            1 if operation.type == 'ConvTranspose' else 0)
                    conv_weight_config.observer_algorithm = 'minmax'

            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = OQC.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        (QuantizationProperty.PER_CHANNEL if
                         self.per_channel else QuantizationProperty.PER_TENSOR)
                    )
                    gemm_weight_config.num_of_bits = self.bit_w
                    gemm_weight_config.quant_min = -pow(2, self.bit_w - 1)
                    gemm_weight_config.quant_max = pow(2,
                                                       self.bit_w - 1) - 1
                    if self.per_channel:
                        gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = 'minmax'

            if operation.num_of_input > 2:
                bias_config = OQC.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type == 'LayerNormalization':
            # Layernorm - gamma and beta need to be FP32
            for TQC in OQC.input_quantization_config[1:]:
                TQC.state = QuantizationStates.FP32

        if operation.type == 'Attention':
            # Attention - Only input and weight need to be quantized.
            for TQC in OQC.input_quantization_config[2:]:
                TQC.state = QuantizationStates.FP32

        return OQC

    @property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.TRT_INT8

    @property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool', 'Softmax',
            'Mul', 'Add', 'Max', 'Sub', 'Div', 'Reshape',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Interp',
            'ReduceMean', 'Transpose', 'Slice', 'Flatten',
            'HardSwish', 'HardSigmoid', 'MatMul',
            'Attention', 'LayerNormalization', 'Gelu',
            'PPQBiasFusedMatMul'
        }

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip', 'Swish', 'Clip', 'SoftPlus', 'Sigmoid', 'Gelu'}
