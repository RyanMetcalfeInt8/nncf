import unittest
import numpy as np
import onnx
import nncf
from onnx.reference import ReferenceEvaluator
from onnx import helper as h, numpy_helper as nh


class DecompressionTests(unittest.TestCase):
    def model(self, opset=21):
        return h.make_model(h.make_graph([h.make_node('Transpose',['w'],['wt'],perm=[1,0]),
            h.make_node('MatMul',['x','wt'],['y'])],'g',[h.make_tensor_value_info('x',1,[2,16])],
            [h.make_tensor_value_info('y',1,[2,8])],[nh.from_array(np.random.default_rng(4).normal(size=(8,16)).astype(np.float32),'w')]),
            opset_imports=[h.make_opsetid('',opset)],ir_version=10)

    def compress(self, dtype=None, opset=21):
        return nncf.compress_weights(self.model(opset),mode=nncf.CompressWeightsMode.INT8_SYM,
            advanced_parameters=nncf.AdvancedCompressionParameters(onnx_decompression_dtype=dtype))

    def test_default(self):
        m=self.compress();onnx.checker.check_model(m)
        self.assertFalse(any(n.op_type=='Cast' for n in m.graph.node))
        self.assertEqual(next(t for t in m.graph.initializer if t.name.endswith('_scale')).data_type,onnx.TensorProto.FLOAT)

    def test_float16(self):
        m=self.compress('float16');onnx.checker.check_model(m)
        self.assertEqual(next(t for t in m.graph.initializer if t.name.endswith('_scale')).data_type,onnx.TensorProto.FLOAT16)
        cast=next(n for n in m.graph.node if n.op_type=='Cast')
        self.assertEqual(next(a.i for a in cast.attribute if a.name=='to'),onnx.TensorProto.FLOAT)
        dq=next(n for n in m.graph.node if n.op_type=='DequantizeLinear')
        self.assertEqual(cast.input[0],dq.output[0])
        out=ReferenceEvaluator(m).run(None,{'x':np.ones((2,16),np.float32)})[0]
        self.assertEqual(out.dtype,np.float32);self.assertTrue(np.isfinite(out).all())

    def test_invalid(self):
        with self.assertRaises(nncf.ParameterNotSupportedError):self.compress('float32')

    def test_old_opset(self):
        with self.assertRaises(nncf.ParameterNotSupportedError):self.compress('float16',13)


if __name__=='__main__':unittest.main()
