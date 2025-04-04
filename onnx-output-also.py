import onnx
from onnx import helper
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model

import typer
from typing_extensions import Annotated

import sys

def log(x):
   sys.stderr.write('# ' + str(x) + '\n')

alsoapp = typer.Typer(no_args_is_help=True, add_completion=False)

@alsoapp.command()
def main(in_model: Annotated[str, typer.Argument(help="onnx input model")],
         tensor_name: Annotated[str, typer.Argument(help="tensor name to add to outputs")] = "",
         out_model: Annotated[str, typer.Argument(help="onnx output model (if empty, same as in_model)")] = "",
         list_tensors: bool = typer.Option(False, "--list", "-l", help="list tensors (do not write out_model)"),
):
    log("loading %s" % in_model)
    model = load_onnx_model(in_model)
    graph = model.graph
    if list_tensors:
        for name in enumerate_model_node_outputs(model):
            print(name)
        log(graph.value_info)
        for value_info_proto in graph.value_info:
            log(value_info_proto.name)
        return
    if not len(tensor):
        log("specify tensorname [onnx-outfile] or -l")
        return False
    value_info = None
    for value_info_proto in graph.value_info:
        if value_info_proto.name == tensor_name:
            value_info = value_info_proto
            break
    if value_info:
        graph.output.append(intermediate_layer_value_info)
    if not len(out_model):
        out_model = in_model
    if not onnx.checker.check_model(model):
        log("model checker failed")
        return
    save_onnx_model(model, out_model)

if __name__ == "__main__":
    alsoapp()
