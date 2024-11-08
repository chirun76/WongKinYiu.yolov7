import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper
from onnx import TensorProto

def create_constant_initializer(output_name, tensor):
    """Create an initializer tensor"""
    return numpy_helper.from_array(tensor, name=output_name)

def create_constant_node(output_name, tensor):
    # Create Constant node
    return helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=[output_name],
        name='Constant__replaced_{}'.format(output_name),
        value=tensor)

def create_reshape_node(output_name, input_name, shape):
    # Create Reshape node
    shape_t_name = "{}__shape".format(output_name)
    reshape_node = helper.make_node(
        op_type='Reshape',
        inputs=[input_name, shape_t_name],
        outputs=[output_name],
        name='Reshape__replaced_{}'.format(output_name))
    # Create shape Constant node
    shape_arr = np.array(shape)
    shape_tensor = numpy_helper.from_array(shape_arr, name=shape_t_name)
    shape_node = create_constant_node(shape_t_name, shape_tensor)
    return [shape_node, reshape_node]

def create_transpose_node(output_name, input_name, perm):
    # Create Transpose node
    transpose_node = helper.make_node(
        op_type='Transpose',
        inputs=[input_name],
        outputs=[output_name],
        name='Transpose__replaced_{}'.format(output_name),
        perm=perm)
    return transpose_node

def create_pow_node(output_name, input_name_x, exponent_name):
    pow_node = helper.make_node(
        op_type='Pow',
        inputs=[input_name_x, exponent_name],
        outputs=[output_name],
        name='/model/model.77/Pow_2'
    )
    return pow_node

def create_split_node(output_names, input_name):
    # Create Split node with the 'split' attribute specifying the split sizes
    split_node = helper.make_node(
        op_type='Split',
        inputs=[input_name],
        outputs=output_names,
        name='/model/model.77/Split_2',
        axis=3,  # Specify the axis to split along
        split=[2, 2, 81]  # Specify how to split the last dimension
    )
    return split_node

def create_constant_4_initializer(output_name):
    values = []
    for i in range(20):
        row = []
        for j in range(20):
            first_col_value = (j * 32) - 16
            second_col_value = (i * 32) - 16
            row.append([first_col_value, second_col_value])
        values.append(row)
    values = np.array(values, dtype=np.float32).reshape([1, 1, 400, 2])
    return numpy_helper.from_array(values, name=output_name)

def create_pow_initializer(output_name):
    return numpy_helper.from_array(np.array(2, dtype=np.float32), name=output_name)

def create_reshape_initializer(output_name, shape):
    shape_arr = np.array(shape, dtype=np.int64)
    return numpy_helper.from_array(shape_arr, name=output_name)

def create_test_output_graph(new_graph_name):
    print("====== Creating Conv_output_0-to-Reshape_5_output_0 Replacement Graph with Mul and Add Nodes ======")
    
    new_nodes = []
    initializers = []

    new_inputs = [helper.make_tensor_value_info(
        '/model/model.77/m.2/Conv_output_0', TensorProto.FLOAT, [1, 255, 20, 20])]
    new_outputs = [helper.make_tensor_value_info(
        '/model/model.77/Reshape_5_output_0', TensorProto.FLOAT, [1, 1200, 85])]

    new_nodes.extend(create_reshape_node('/model/model.77/Reshape_4_output_0', '/model/model.77/m.2/Conv_output_0', [1, 3, 85, 400]))
    new_nodes.append(create_transpose_node('/model/model.77/Transpose_2_output_0', '/model/model.77/Reshape_4_output_0', [0,1,3,2]))

    sigmoid_node = helper.make_node(
        op_type='Sigmoid',
        inputs=['/model/model.77/Transpose_2_output_0'],
        outputs=['/model/model.77/Sigmoid_2_output_0'],
        name='/model/model.77/Sigmoid_2'
    )
    new_nodes.append(sigmoid_node)

    split_output_names = ['/model/model.77/Split_2_output_0', '/model/model.77/Split_2_output_1', '/model/model.77/Split_2_output_2']
    split_node = create_split_node(split_output_names, '/model/model.77/Sigmoid_2_output_0')
    new_nodes.append(split_node)

    constant_tensor = numpy_helper.from_array(np.array(64, dtype=np.float32), name='/model/model.77/Constant_15_output_0')
    initializers.append(constant_tensor)

    mul_node = helper.make_node( 
        op_type='Mul',
        inputs=['/model/model.77/Split_2_output_0', '/model/model.77/Constant_15_output_0'],
        outputs=['/model/model.77/Mul_4_output_0'],
        name='/model/model.77/Mul_4'
    )
    new_nodes.append(mul_node)

    add_constant_tensor = create_constant_4_initializer('/model/model.77/Constant_16_output_0')
    initializers.append(add_constant_tensor)

    add_node = helper.make_node(
        op_type='Add',
        inputs=['/model/model.77/Mul_4_output_0', '/model/model.77/Constant_16_output_0'],
        outputs=['/model/model.77/Add_2_output_0'],
        name='/model/model.77/Add_2'
    )
    new_nodes.append(add_node)

    pow_initializer = create_pow_initializer('/model/model.77/Constant_5_output_0')
    initializers.append(pow_initializer)

    pow_node = create_pow_node('/model/model.77/Pow_2_output_0', '/model/model.77/Split_2_output_1', '/model/model.77/Constant_5_output_0')
    new_nodes.append(pow_node)

    constant_6_tensor = create_constant_initializer('/model/model.77/Constant_18_output_0', np.array([[[[568, 440]], [[768, 972]], [[1836, 1604]]]], dtype=np.float32))
    initializers.append(constant_6_tensor)

    mul_node_after_pow = helper.make_node(
        op_type='Mul',
        inputs=['/model/model.77/Pow_2_output_0', '/model/model.77/Constant_18_output_0'],
        outputs=['/model/model.77/Mul_5_output_0'],
        name='/model/model.77/Mul_5'
    )
    new_nodes.append(mul_node_after_pow)

    concat_node = helper.make_node(
        op_type='Concat',
        inputs=['/model/model.77/Add_2_output_0', '/model/model.77/Mul_5_output_0', '/model/model.77/Split_2_output_2'],
        outputs=['/model/model.77/Concat_2_output_0'],
        name='/model/model.77/Concat_2',
        axis=3
    )
    new_nodes.append(concat_node)

    new_nodes.extend(create_reshape_node('/model/model.77/Reshape_5_output_0', '/model/model.77/Concat_2_output_0', [1, -1, 85]))

    # Create the new graph
    new_graph = helper.make_graph(
        new_nodes,
        new_graph_name,
        new_inputs,
        new_outputs,
        initializer=initializers,
    )
    
    new_model = helper.make_model(new_graph, producer_name='onnx-example', opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(new_model)
    inferred_model = onnx.shape_inference.infer_shapes(new_model)
    onnx.save(inferred_model, '{}.onnx'.format(new_graph_name))
    print(f"Model '{new_graph_name}.onnx' created successfully.")

if __name__ == '__main__':
    create_test_output_graph('replace_subgraph_3')
