# Optional ONNX weight decompression precision

```python
compressed = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT8_SYM,
    advanced_parameters=nncf.AdvancedCompressionParameters(
        onnx_decompression_dtype="float16",
    ),
)
```

The option defaults to None, retaining the existing ONNX compression behavior.
When enabled, NNCF writes f16 scales and emits DequantizeLinear producing f16,
then Cast back to the original weight dtype when necessary. Existing activation
and MatMul types remain unchanged. Scales and dequantization arithmetic incur
f16 rounding; validate model accuracy and runtime support before adoption.

The option is implemented by the ONNX backend and requires opset 19 or later.
It does not apply to OpenVINO compression, whose decompression already uses f16.
Only "float16" is currently accepted. The older-opset MatMulNBits route is
explicitly rejected when this option is requested. This change does not select
the new option automatically in the workspace Mobius helper.

Validation includes default behavior, f16 scale and Cast structure, ONNX
reference execution, invalid values and older opsets. OpenVINO frontend conversion
produces f16 Convert/Multiply followed by f32 Convert for f32 source weights.
Full-model NPU performance and representative accuracy remain to be validated.
