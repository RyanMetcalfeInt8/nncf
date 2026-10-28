# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper
from onnx import numpy_helper

import nncf
from nncf.common.graph.graph import get_node_names_matching_graph_pattern
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.quantization.ignored_patterns import create_rope


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("weight_port", [0, 1])
@pytest.mark.parametrize("decompression_dtype", ["float16", None])
def test_rope_frequencies_remain_float_during_weight_compression(copies, weight_port, decompression_dtype):
    """Keep phase frequencies intact while unrelated learned weights are compressed."""
    frequencies = np.geomspace(1, 1e-6, 32).astype(np.float32)
    frequency_shape = (1, 32, 1) if weight_port == 0 else (1, 1, 32)
    position_shape = [1, 1, 16] if weight_port == 0 else [1, 16, 1]
    mm_inputs = ["inv_freq", "positions"] if weight_port == 0 else ["positions", "inv_freq"]
    rng = np.random.default_rng(42)
    weights = rng.normal(size=(32, 32)).astype(np.float32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", mm_inputs, ["angles"], name="rope_matmul"),
                helper.make_node("Transpose", ["angles"], ["transposed"], perm=[0, 2, 1]),
                helper.make_node("Concat", ["transposed"] * copies, ["duplicated"], axis=-1),
                helper.make_node("Cos", ["duplicated"], ["cos"]),
                helper.make_node("Sin", ["duplicated"], ["sin"]),
                helper.make_node("MatMul", ["hidden", "weights"], ["projection"], name="learned_matmul"),
            ],
            "rope_with_projection",
            [
                helper.make_tensor_value_info("positions", onnx.TensorProto.FLOAT, position_shape),
                helper.make_tensor_value_info("hidden", onnx.TensorProto.FLOAT, [1, 32]),
            ],
            [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [None, None, None]) for name in ("cos", "sin")]
            + [helper.make_tensor_value_info("projection", onnx.TensorProto.FLOAT, [1, 32])],
            [
                numpy_helper.from_array(frequencies.reshape(frequency_shape), "inv_freq"),
                numpy_helper.from_array(weights, "weights"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    graph = GraphConverter.create_nncf_graph(model)
    ignored = get_node_names_matching_graph_pattern(graph, create_rope())
    assert "rope_matmul" in ignored
    assert "learned_matmul" not in ignored
    compressed = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        group_size=-1,
        all_layers=True,
        advanced_parameters=nncf.AdvancedCompressionParameters(onnx_decompression_dtype=decompression_dtype),
    )
    constants = {tensor.name: tensor for tensor in compressed.graph.initializer}
    assert constants["inv_freq"].data_type == onnx.TensorProto.FLOAT
    np.testing.assert_array_equal(numpy_helper.to_array(constants["inv_freq"]), frequencies.reshape(frequency_shape))
    assert "weights" not in constants
    assert any(tensor.data_type in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4) for tensor in constants.values())
    for start in (0, 128, 2048, 4096):
        feeds = {
            "positions": np.arange(start, start + 16, dtype=np.float32).reshape(position_shape),
            "hidden": rng.normal(size=(1, 32)).astype(np.float32),
        }
        expected = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            ["cos", "sin"], feeds
        )
        actual = ort.InferenceSession(compressed.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            ["cos", "sin"], feeds
        )
        for reference, output in zip(expected, actual):
            np.testing.assert_array_equal(output, reference)
