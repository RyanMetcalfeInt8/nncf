"""Run with the patched NNCF copy first on PYTHONPATH."""
import unittest
import numpy as np
import nncf
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator


class TransposeAxisTests(unittest.TestCase):
    def check_case(self, out_features, transposes, default_perm=False, gemm=False):
        rng = np.random.default_rng(14)
        w = rng.normal(size=(out_features,16)).astype(np.float32)
        x = rng.normal(size=(2,16)).astype(np.float32)
        nodes = []; rhs = 'w'
        for i in range(transposes):
            out = f't{i}'
            nodes.append(h.make_node('Transpose',[rhs],[out],name=out,
                                     **({} if default_perm else {'perm':[1,0]})))
            rhs = out
        if gemm:
            nodes.append(h.make_node('Gemm',['x',rhs],['y'],name='linear',transB=1))
        else:
            nodes.append(h.make_node('MatMul',['x',rhs],['y'],name='linear'))
        # Gemm with transB=1 requires an even number of explicit transposes.
        effective_transpose = (transposes % 2 == 1) != gemm
        stored = w if effective_transpose else w.T.copy()
        model = h.make_model(h.make_graph(nodes,'g',[h.make_tensor_value_info('x',1,[2,16])],
            [h.make_tensor_value_info('y',1,[2,out_features])],[nh.from_array(stored,'w')]),
            opset_imports=[h.make_opsetid('',21)],ir_version=10)
        compressed = nncf.compress_weights(model,mode=nncf.CompressWeightsMode.INT8_SYM)
        onnx.checker.check_model(compressed)
        dq = next(n for n in compressed.graph.node if n.op_type=='DequantizeLinear')
        constants = {t.name:t for t in compressed.graph.initializer}
        self.assertEqual(list(constants[dq.input[1]].dims),[out_features])
        axis = next((a.i for a in dq.attribute if a.name=='axis'),1) % 2
        self.assertEqual(axis,0 if effective_transpose else 1)
        self.assertTrue(np.isfinite(ReferenceEvaluator(compressed).run(None,{'x':x})[0]).all())

    def test_direct(self): self.check_case(8,0)
    def test_transpose(self): self.check_case(8,1)
    def test_square(self): self.check_case(16,1)
    def test_two_transposes(self): self.check_case(8,2)
    def test_default_perm(self): self.check_case(8,1,default_perm=True)
    def test_gemm(self): self.check_case(8,0,gemm=True)
    def test_gemm_transpose(self): self.check_case(8,1,gemm=True)


if __name__=='__main__': unittest.main()
