echo "I bought a new car because I was going through a midlife crisis . ||| Compré un auto nuevo porque estaba pasando por una crisis de la mediana edad ." > engspa.txt
python awesome_align/run_align.py --output_file=align.engspa.txt --model_name_or_path=--model_name_or_path=bert-base-multilingual-cased --data_file=engspa.txt --extraction='softmax' --softmax_threshold=1e-3 --batch_size=32
