"""Per-channel DQ must use 1D scales even when weights have several channel axes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
from onnx import helper as h, numpy_helper as nh
from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXWeightCompressionAlgoBackend


class MultidimensionalChannelScales(unittest.TestCase):
    def test_embedding_and_depthwise_shapes(self):
        for shape, scale_shape in [((2, 7, 8), (2, 7)), ((8, 1, 5), (8, 1))]:
            for f16 in [False, True]:
                with self.subTest(shape=shape, f16=f16):
                    weights = np.arange(np.prod(shape), dtype=np.int8).reshape(shape)
                    scales = np.linspace(.01, .1, np.prod(scale_shape), dtype=np.float32).reshape(scale_shape)
                    model = h.make_model(h.make_graph(
                        [h.make_node('Identity', ['w'], ['y'])], 'g', [],
                        [h.make_tensor_value_info('y', 1, list(shape))],
                        [nh.from_array(weights.astype(np.float32), 'w')]),
                        opset_imports=[h.make_opsetid('', 24)], ir_version=10)
                    backend = ONNXWeightCompressionAlgoBackend(model)
                    backend._add_dequantize_linear_layer(
                        model, weights, scales, None, 0, 0, 'w', onnx.TensorProto.INT8,
                        scale_dtype=10 if f16 else 1, output_dtype=1 if f16 else None)
                    onnx.checker.check_model(model)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'm.onnx'
                        onnx.save(model, path)
                        actual = next(iter(ov.Core().compile_model(str(path), 'CPU')({}).values()))
                    dtype = np.float16 if f16 else np.float32
                    expected = weights.astype(dtype) * scales.astype(dtype)[..., None]
                    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=.002, atol=.002)


if __name__ == '__main__':
    unittest.main()
