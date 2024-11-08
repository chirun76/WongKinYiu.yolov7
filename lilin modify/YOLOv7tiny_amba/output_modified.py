import onnx
from onnx import helper

model = onnx.load('version2.onnx')
new_graph = model.graph
target_node_name = '/model/model.77/Concat_3'

# Find target_node_name
target_node_index = None
for i, node in enumerate(new_graph.node):
    if target_node_name in node.name:
        target_node_index = i
        break

if target_node_index is not None:
    # Delete all nodes after target_node_name
    nodes_to_remove = new_graph.node[target_node_index + 1:]
    for node in nodes_to_remove:
        new_graph.node.remove(node)

    # Get the output name of the target node
    concat_output_name = new_graph.node[target_node_index].output[0]
    
    # Remove original output and add new output
    for output in new_graph.output[:]:
        new_graph.output.remove(output)
    
    new_graph.output.append(helper.make_tensor_value_info(concat_output_name, onnx.TensorProto.FLOAT, [1, 25200, 85]))  

    inferred_model = onnx.shape_inference.infer_shapes(model)
    try:
        onnx.checker.check_model(inferred_model)
        print("Model check passed.")
        onnx.save(inferred_model, 'version3.onnx')
    except onnx.checker.ValidationError as e:
        print("Model check failed.")
        print(e)
else:
    print("Target node not found.")