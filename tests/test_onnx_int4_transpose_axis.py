"""Regression: blockwise DQ must use the grouped source-weight dimension."""
import unittest
import numpy as np
import nncf
import onnx
import openvino as ov
from onnx import helper as h, numpy_helper as nh


class Int4AxisTests(unittest.TestCase):
    def check_case(self, transpose, f16):
        w=np.random.default_rng(7).normal(size=(32,64)).astype(np.float32)
        nodes=[];rhs='w'
        if transpose:
            nodes.append(h.make_node('Transpose',['w'],['wt'],perm=[1,0]));rhs='wt'
        nodes.append(h.make_node('MatMul',['x',rhs],['y'],name='linear'))
        m=h.make_model(h.make_graph(nodes,'g',[h.make_tensor_value_info('x',1,[2,64])],
            [h.make_tensor_value_info('y',1,[2,32])],[nh.from_array(w if transpose else w.T.copy(),'w')]),
            opset_imports=[h.make_opsetid('',24)],ir_version=10)
        m=nncf.compress_weights(m,mode=nncf.CompressWeightsMode.INT4_SYM,group_size=16,ratio=1,
            all_layers=True,advanced_parameters=nncf.AdvancedCompressionParameters(
                onnx_decompression_dtype='float16' if f16 else None))
        onnx.checker.check_model(m)
        dq=next(n for n in m.graph.node if n.op_type=='DequantizeLinear')
        attrs={a.name:h.get_attribute_value(a) for a in dq.attribute}
        self.assertEqual(attrs['axis'],1 if transpose else 0)
        cs={t.name:t for t in m.graph.initializer}
        self.assertEqual(list(cs[dq.input[1]].dims),[32,4] if transpose else [4,32])
        # Conversion checks block-size/shape compatibility, unlike ONNX checker.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.onnx';onnx.save(m,path)
            core=ov.Core();converted=core.read_model(str(path))
            out=next(iter(core.compile_model(converted,'CPU')({0:np.ones((2,64),np.float32)}).values()))
            self.assertTrue(np.isfinite(out).all());self.assertEqual(out.shape,(2,32))

    def test_transpose_f16(self):self.check_case(True,True)
    def test_direct_f16(self):self.check_case(False,True)
    def test_transpose_default(self):self.check_case(True,False)
    def test_direct_default(self):self.check_case(False,False)


if __name__=='__main__':unittest.main()
