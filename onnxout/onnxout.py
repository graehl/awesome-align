import onnx

import typer
from typing_extensions import Annotated

import sys

def log(x):
   sys.stderr.write('# ' + str(x) + '\n')

onnxoutapp = typer.Typer(no_args_is_help=True, add_completion=False)

def onnx_inputs(path, inputs=None):
  if inputs is None:
    return [x.name for x in onnx.load(path).graph.input]
  return inputs

def modify_onnx_outputs(path, onnxpathout, outputs, inputs=None, checker=True):
  onnx.utils.extract_model(path, onnxpathout, onnx_inputs(path, inputs), outputs)
  if checker:
    onnx.checker.check_model(onnxpathout)
  return onnxpathout


@onnxoutapp.command()
def main(in_model: Annotated[str, typer.Argument(help="onnx input model")],
         name: Annotated[str, typer.Argument(help="tensor name to outputs")] = "",
         out_model: Annotated[str, typer.Argument(help="onnx output model (if empty, same as in_model)")] = "",
         list_tensors: bool = typer.Option(False, "--list", "-l", help="list tensors (do not write out_model)"),
):
    log("loading %s" % in_model)
    if list_tensors:
        model = onnx.load(in_model)
        graph = model.graph
        for name in enumerate_model_node_outputs(model):
            print(name)
        log(graph.value_info)
        for value_info_proto in graph.value_info:
            log(value_info_proto.name)
        return
    if not len(name):
        log("specify tensorname [onnx-outfile] or -l")
        return False
    modify_onnx_outputs(in_model, out_model, [name], inputs=None, checker=True)

if __name__ == "__main__":
    onnxoutapp()
