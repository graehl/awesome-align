#!/usr/bin/env python3
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
         name: Annotated[str, typer.Argument(help="tensor name to outputs")] = "/layer.7/output/LayerNorm/Add_1_output_0",
         out_model: Annotated[str, typer.Argument(help="onnx output model (if empty, same as in_model)")] = "",
         list_tensors: bool = typer.Option(False, "--list", "-l", help="list tensors (do not write out_model)"),
         startswith: str = typer.Option(None, "--startswith", "-s", help="list tensors starting with this (do not write out_model)"),
):
    log("loading %s" % in_model)
    if list_tensors or startswith:
        log("listing tensors in %s" % in_model)
        model = onnx.load(in_model)
        log("loaded %s" % in_model)
        for x in model.graph.input:
           log(f'graph input {x.name}')
        for x in model.graph.output:
           log(f'graph output {x.name}')
        seen = set()
        for n in model.graph.node:
           if n.name not in seen:
              seen.add(n.name)
              for oname in n.output:
                 if startswith is None or oname.startswith(startswith):
                    print(f'{oname}')
        return
    if not len(name):
        log("specify tensorname [onnx-outfile] or -l")
        return False
    modify_onnx_outputs(in_model, out_model, [name], inputs=None, checker=True)

if __name__ == "__main__":
    onnxoutapp()
