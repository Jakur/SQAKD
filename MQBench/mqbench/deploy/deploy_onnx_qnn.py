import onnx
import numpy as np

from mqbench.utils.logger import logger
from .common import ONNXGraph
from mqbench.deploy.common import  parse_attrs

FAKE_QUANTIZE_OP = ['QuantizeLinear', 'DequantizeLinear']


def search_and_replace_input(next_node, name, new_name):
    for idx, _input_name in enumerate(next_node.input):
        if _input_name == name:
            next_node.input[idx] = new_name

class ONNXQNNPass(object):
    def __init__(self, onnx_model_path):
        self.onnx_model = ONNXGraph(onnx_model_path)

    @property
    def qlinear_op_type(self):
        return ['QLinearConv', 'QLinearAdd', 'QLinearGemm', 'QLinearGlobalAveragePool',
                'QLinearAveragePool', 'QLinearConcat']

    @staticmethod
    def attribute_to_kwarg(attribute):
        '''
        Convert attribute to kwarg format for use with onnx.helper.make_node.
            :parameter attribute: attribute in AttributeProto format.
            :return: attribute in {key: value} format.
        '''
        if (attribute.type == 0):
            raise ValueError('attribute {} does not have type specified.'.format(attribute.name))

        # Based on attribute type definitions from AttributeProto
        # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
        if (attribute.type == 1):
            value = attribute.f
        elif (attribute.type == 2):
            value = attribute.i
        elif (attribute.type == 3):
            value = attribute.s
        elif (attribute.type == 4):
            value = attribute.t
        elif (attribute.type == 5):
            value = attribute.g
        elif (attribute.type == 6):
            value = attribute.floats
        elif (attribute.type == 7):
            value = attribute.ints
        elif (attribute.type == 8):
            value = attribute.strings
        elif (attribute.type == 9):
            value = attribute.tensors
        elif (attribute.type == 10):
            value = attribute.graphs
        else:
            raise ValueError('attribute {} has unsupported type {}.'.format(attribute.name, attribute.type))

        return {attribute.name: value}

    def quantize_weight(self, weight_name, scale_name, zero_point_name):
        weight = self.onnx_model.get_initializer(weight_name)
        scale = self.onnx_model.get_initializer(scale_name)
        zero_point = self.onnx_model.get_initializer(zero_point_name)
        return ((weight / scale).round() + zero_point).astype(np.uint8)

    def quantize_bias(self, bias, x_scale, w_scale):
        x_scale = self.onnx_model.get_initializer(x_scale)
        w_scale = self.onnx_model.get_initializer(w_scale)
        bias = self.onnx_model.get_initializer(bias)
        return (bias / (x_scale * w_scale)).astype(np.int32)

    @property
    def node_without_qparams(self):
        return ['Flatten']

    def replace_conv_gemm(self, node, idx, is_conv):
        # Input scale
        qlinear_conv_inputs = []
        input_fake_dequant_node = self.onnx_model.get_tensor_producer(node.input[0])
        input_fake_quant_node = self.onnx_model.get_tensor_producer(input_fake_dequant_node.input[0])
        assert input_fake_quant_node.op_type == 'QuantizeLinear'
        x_scale, x_zero_point = input_fake_quant_node.input[1], input_fake_quant_node.input[2]
        # Output scale
        node_next_quant = self.onnx_model.get_tensor_consumer(node.output[0])[0]
        # node_next_dequant = self.onnx_model.get_tensor_consumer(node_next_quant.output[0])[0]
        qlinear_conv_output = node.output
        y_scale, y_zero_point = self.get_node_output_qparams(node)
        # Weight scale
        weight_fake_dequant_node = self.onnx_model.get_tensor_producer(node.input[1])
        weight_fake_quant_node = self.onnx_model.get_tensor_producer(weight_fake_dequant_node.input[0])
        w_scale, w_zero_point = weight_fake_quant_node.input[1], weight_fake_quant_node.input[2]
        weight_name = weight_fake_quant_node.input[0]
        W = self.quantize_weight(weight_name, w_scale, w_zero_point)
        self.onnx_model.set_initializer(weight_name, W)
        qlinear_conv_inputs.extend([node.input[0], x_scale, x_zero_point,
                                    weight_name, w_scale, w_zero_point,
                                    y_scale, y_zero_point])
        # Bias
        if len(node.input) == 3:
            bias_name = node.input[2]
            B = self.quantize_bias(bias_name, x_scale, w_scale)
            self.onnx_model.set_initializer(bias_name, B)
            qlinear_conv_inputs.append(bias_name)
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        node_type = "QLinearConv" if is_conv else "QLinearGemm"
        qlinear_conv_node = onnx.helper.make_node(node_type, 
                                                  qlinear_conv_inputs,
                                                  qlinear_conv_output,
                                                  node.name + '_quantized',
                                                  **kwargs)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.remove_node_purely(weight_fake_quant_node)
        self.onnx_model.remove_node_purely(weight_fake_dequant_node)
        # self.onnx_model.remove_node_purely(node_next_quant)
        # next_nodes = self.onnx_model.get_tensor_consumer(input_fake_dequant_node.output[0])
        # for next_node in next_nodes:
        #     search_and_replace_input(next_node, input_fake_dequant_node.output[0], input_fake_quant_node.output[0])
        # self.onnx_model.remove_node_purely(input_fake_dequant_node)
        self.onnx_model.insert_node_purely(qlinear_conv_node, idx)
        self.onnx_model.topologize_graph()

    def replace_add_to_qlinearadd(self, node, idx):
        # First input
        qlinear_add_input = []
        qlinear_add_output = node.output
        first_input_dequant_node = self.onnx_model.get_tensor_producer(node.input[0])
        first_input_quant_node = self.onnx_model.get_tensor_producer(first_input_dequant_node.input[0])
        assert first_input_quant_node.op_type == 'QuantizeLinear'
        first_input_quantized = first_input_dequant_node.output[0]
        first_scale = first_input_quant_node.input[1]
        first_zero_point = first_input_quant_node.input[2]
        # Second input
        second_input_dequant_node = self.onnx_model.get_tensor_producer(node.input[1])
        second_input_quant_node = self.onnx_model.get_tensor_producer(second_input_dequant_node.input[0])
        assert second_input_quant_node.op_type == 'QuantizeLinear'
        second_input_quantized = second_input_dequant_node.output[0]
        second_scale = second_input_quant_node.input[1]
        second_zero_point = second_input_quant_node.input[2]
        # Output
        output_scale, output_zero_point = self.get_node_output_qparams(node)
        qlinear_add_input.extend([first_input_quantized, first_scale, first_zero_point,
                                  second_input_quantized, second_scale, second_zero_point,
                                  output_scale, output_zero_point])
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_add_node = onnx.helper.make_node("QLinearAdd", 
                                                 qlinear_add_input,
                                                 qlinear_add_output,
                                                 node.name + '_quantized',
                                                 domain='com.microsoft',
                                                 **kwargs)
        self.onnx_model.insert_node_purely(qlinear_add_node, idx)
        self.onnx_model.remove_node_purely(node)
        # first_next_nodes = self.onnx_model.get_tensor_consumer(first_input_dequant_node.output[0])
        # for next_node in first_next_nodes:
        #     search_and_replace_input(next_node, first_input_dequant_node.output[0], first_input_quant_node.output[0])
        # second_next_nodes = self.onnx_model.get_tensor_consumer(second_input_dequant_node.output[0])
        # for next_node in second_next_nodes:
        #     search_and_replace_input(next_node, second_input_dequant_node.output[0], second_input_quant_node.output[0])
        # self.onnx_model.remove_node_purely(first_input_dequant_node)
        # self.onnx_model.remove_node_purely(second_input_dequant_node)
        self.onnx_model.topologize_graph()

    def replace_pool_to_qlinearpool(self, node, idx, is_global):
        qlinear_pool_input = []
        prev_dequant_node = self.onnx_model.get_tensor_producer(node.input[0])
        prev_quant_node = self.onnx_model.get_tensor_producer(prev_dequant_node.input[0])
        assert prev_quant_node.op_type == 'QuantizeLinear'
        x_scale, x_zero_point = prev_quant_node.input[1], prev_quant_node.input[2]
        y_scale, y_zero_point = self.get_node_output_qparams(node)
        qlinear_pool_input.extend([node.input[0], x_scale, x_zero_point,
                                   y_scale, y_zero_point])
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_add_output = node.output
        node_type = "QLinearGlobalAveragePool" if is_global else "QLinearAveragePool"
        qlinear_pool_node = onnx.helper.make_node(node_type, 
                                                  qlinear_pool_input,
                                                  qlinear_add_output,
                                                  node.name + '_quantized',
                                                  domain='com.microsoft',
                                                  **kwargs)
        # next_nodes = self.onnx_model.get_tensor_consumer(prev_dequant_node.output[0])
        # for next_node in next_nodes:
        #     search_and_replace_input(next_node, prev_dequant_node.output[0], prev_quant_node.output[0])
        self.onnx_model.insert_node_purely(qlinear_pool_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def get_node_output_qparams(self, node):
        fake_quantize_node = self.onnx_model.get_tensor_consumer(node.output[0])[0]
        while fake_quantize_node.op_type != 'QuantizeLinear':
            assert fake_quantize_node.op_type in self.node_without_qparams
            fake_quantize_node = self.onnx_model.get_tensor_consumer(fake_quantize_node.output[0])[0]
        return fake_quantize_node.input[1], fake_quantize_node.input[2]

    def replace_op_pass(self):
        # Replace Conv / Gemm / Add / AvgPool / Concat / LeakyRelu.
        op_types = set()
        for idx, node in enumerate(self.onnx_model.graph.node):
            op_types.add(node.op_type)
            if node.op_type == 'Conv':
                self.replace_conv_gemm(node, idx, is_conv=True)
            if node.op_type == 'Gemm':
                pass
                # onnxruntime and tvm is not supported yet.
                # self.replace_conv_gemm(node, idx, is_conv=False)
            if node.op_type == 'Add':
                self.replace_add_to_qlinearadd(node, idx)
            if node.op_type == 'GlobalAveragePool':
                self.replace_pool_to_qlinearpool(node, idx, is_global=True)
            if node.op_type == 'AveragePool':
                self.replace_pool_to_qlinearpool(node, idx, is_global=False)
            # TODO
            if node.op_type == 'Concat':
                pass
            if node.op_type == 'LeakyRelu':
                pass

    # def replace_qlinear_layer_pass(self):
    #     # Replace FakeQuantize
    #     remove_nodes = []
    #     for node in self.onnx_model.graph.node:
    #         if node.op_type in FAKE_QUANTIZE_OP:
    #             prev_node = self.onnx_model.get_tensor_producer(node.input[0])
    #             next_node_list = self.onnx_model.get_tensor_consumer(node.output[0])
    #             quantize_node = None
    #             dequantize_node = None
    #             output_flag = False
    #             for next_node in next_node_list:
    #                 if prev_node != 'INPUT_TOKEN' and prev_node.op_type in self.qlinear_op_type and \
    #                         next_node != 'OUTPUT_TOKEN' and next_node.op_type in self.qlinear_op_type:
    #                     search_and_replace_input(next_node, node.output[0], node.input[0])
    #                 elif prev_node != 'INPUT_TOKEN' and prev_node.op_type in self.qlinear_op_type and \
    #                         next_node == 'OUTPUT_TOKEN':
    #                     if dequantize_node is None:
    #                         output_flag = True
    #                 else:
    #                     if quantize_node is None:
    #                         output_value_info = [f'{node.output[0]}_QuantizeLinear']
    #                         quantize_node = onnx.helper.make_node("QuantizeLinear",
    #                                                               node.input[0:3],
    #                                                               output_value_info,
    #                                                               ('input' if prev_node == 'INPUT_TOKEN' else prev_node.name) + '_quantized')
    #                         self.onnx_model.insert_node_purely(quantize_node)
    #                     search_and_replace_input(next_node, node.output[0], quantize_node.output[0])
    #             if not output_flag:
    #                 self.onnx_model.remove_node_purely(node)
    #                 self.onnx_model.topologize_graph()
    def replace_qlinear_layer_pass(self):
        node_detect = True
        while node_detect:
            node_detect = False
            # Replace FakeQuantize
            for node in self.onnx_model.graph.node:
                if node.op_type in self.qlinear_op_type:
                    next_node_list = self.onnx_model.get_tensor_consumer(node.output[0])
                    for i, next_node in enumerate(next_node_list):
                        if hasattr(next_node, 'op_type'):
                            if next_node.op_type == 'QuantizeLinear':
                                node_detect = True
                                node.output[0] = next_node.output[0]
                                # next_dequant_node_list = self.onnx_model.get_tensor_consumer(next_node.output[0])
                                # for next_dequant_node in next_dequant_node_list:
                                #     search_and_replace_input(next_dequant_node, next_node.output[0], node.output[0])
                                self.onnx_model.remove_node_purely(next_node)
                                self.onnx_model.topologize_graph()
                    for i in range(len(node.input)):
                        pre_node = self.onnx_model.get_tensor_producer(node.input[i])
                        if hasattr(pre_node, 'op_type'):
                            if pre_node.op_type == 'DequantizeLinear':
                                node_detect = True
                                pre_quant_node = self.onnx_model.get_tensor_producer(pre_node.input[0])
                                pre_node_next_list = self.onnx_model.get_tensor_consumer(pre_node.output[0])
                                for pre_node_next_node in pre_node_next_list:
                                    search_and_replace_input(pre_node_next_node, pre_node.output[0], pre_quant_node.output[0])
                                self.onnx_model.remove_node_purely(pre_node)
                                self.onnx_model.topologize_graph()


    def merge_relu_pass(self):
        for node in self.onnx_model.graph.node:
            if node.op_type == 'Relu':
                next_node = self.onnx_model.get_tensor_consumer(node.output[0])[0]
                assert next_node.op_type == 'QuantizeLinear'
                # Input idx2 is zero point.
                self.onnx_model.set_initializer(next_node.input[2], np.array([0], dtype=np.uint8), raw=False)
                self.onnx_model.remove_node_purely(node)
                next_node.input[0] = node.input[0]
            if node.op_type == 'Clip':
                next_node = self.onnx_model.get_tensor_consumer(node.output[0])[0]
                assert next_node.op_type == 'QuantizeLinear'
                # Input idx2 is zero point.
                scale = self.onnx_model.get_initializer(next_node.input[1])
                scale = min(scale, 6.0 / 255)
                self.onnx_model.set_initializer(next_node.input[1], np.array([scale], dtype=np.float32), raw=False)
                self.onnx_model.set_initializer(next_node.input[2], np.array([0], dtype=np.uint8), raw=False)
                self.onnx_model.remove_node_purely(node)
                next_node.input[0] = node.input[0]
        self.onnx_model.topologize_graph()

    def format_qlinear_dtype_pass(self, qmin_max_dict):
        for node in self.onnx_model.graph.node:
            if node.op_type in FAKE_QUANTIZE_OP:
                scale, zero_point = node.input[1], node.input[2]
                qmin, qmax = qmin_max_dict[node.name]
                assert qmax - qmin == 2 ** 8 - 1, "Only 8 bit quantization support deploy to QNN."
                scale_proto = self.onnx_model.initializer[scale][0]
                if scale_proto.raw_data != b'' and scale_proto.dims == []:
                    scale_data = self.onnx_model.get_initializer(scale)
                    self.onnx_model.set_initializer(scale, scale_data.astype(np.float32), raw=False)
                zero_point_proto = self.onnx_model.initializer[zero_point][0]
                zero_point_data = self.onnx_model.get_initializer(zero_point)
                # Align sym and asym scheme.
                zero_point_data = (zero_point_data - qmin).reshape((1,))
                self.onnx_model.set_initializer(zero_point, zero_point_data.astype(np.uint8), raw=False)


    def run(self, model_name, qmin_max_dict):
        self.format_qlinear_dtype_pass(qmin_max_dict)
        self.merge_relu_pass()
        self.replace_op_pass()
        self.replace_qlinear_layer_pass()
        # self.onnx_model.optimize_model()
        self.onnx_model.set_opset_version('com.microsoft', 1)

        # try:
        #     onnx.checker.check_model(self.onnx_model.model)
        # except onnx.checker.ValidationError as e:
        #     logger.critical('The model is invalid: %s' % e)
        self.onnx_model.save_onnx_model('{}.onnx'.format(model_name))
