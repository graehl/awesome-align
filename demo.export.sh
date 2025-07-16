layer=${1:-9}
PYTHONPATH=.:$PYTHONPATH python awesome_align/run_align.py --model_name_or_path=bert-base-multilingual-cased --output_onnx=e${layer}.onnx --max_layer=${layer}
