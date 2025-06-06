# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import random
import itertools
import os
import shutil
import tempfile

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path, offsets=None):
        assert os.path.isfile(file_path)
        print('Loading the dataset...')
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.offsets = offsets

    def process_line(self, worker_id, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None

        src, tgt = line.split(' ||| ')
        if src.rstrip() == '' or tgt.rstrip() == '':
            return None

        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids']
        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt)

    def __iter__(self):
        if self.offsets is not None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id
            offset_start = self.offsets[worker_id]
            offset_end = self.offsets[worker_id+1] if worker_id+1 < len(self.offsets) else None
        else:
            offset_start = 0
            offset_end = None
            worker_id = 0

        with open(self.file_path, encoding="utf-8") as f:
            f.seek(offset_start)
            line = f.readline()
            while line:
                processed = self.process_line(worker_id, line)
                if processed is None:
                    print(f'Line "{line.strip()}" (offset in bytes: {f.tell()}) is not in the correct format. Skipping...')
                    empty_tensor = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                    empty_sent = ''
                    yield (worker_id, empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent)
                else:
                    yield processed
                if offset_end is not None and f.tell() >= offset_end:
                    break
                line = f.readline()

def find_offsets(filename, num_workers):
    if num_workers <= 1:
        return None
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_workers
        offsets = [0]
        for i in range(1, num_workers):
            f.seek(chunk_size * i)
            pos = f.tell()
            while True:
                try:
                    l=f.readline()
                    break
                except UnicodeDecodeError:
                    pos -= 1
                    f.seek(pos)
            offsets.append(f.tell())
    return offsets

def open_writer_list(filename, num_workers):
    writer = open(filename, 'w+', encoding='utf-8')
    writers = [writer]
    if num_workers > 1:
        writers.extend([tempfile.TemporaryFile(mode='w+', encoding='utf-8') for i in range(1, num_workers)])
    return writers

def merge_files(writers):
    if len(writers) == 1:
        writers[0].close()
        return

    for i, writer in enumerate(writers[1:], 1):
        writer.seek(0)
        shutil.copyfileobj(writer, writers[0])
        writer.close()
    writers[0].close()
    return


def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):

    if args.data_file is None:
        if args.output_onnx is None:
            raise ValueError('set data_file or output_onnx')
        return

    def collate(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    offsets = find_offsets(args.data_file, args.num_workers)
    dataset = LineByLineTextDataset(tokenizer, file_path=args.data_file, offsets=offsets)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(0, desc="Extracting")

    writers = open_writer_list(args.output_file, args.num_workers)
    if args.output_prob_file is not None:
        prob_writers = open_writer_list(args.output_prob_file, args.num_workers)
    if args.output_word_file is not None:
        word_writers = open_writer_list(args.output_word_file, args.num_workers)

    for batch in dataloader:
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
            word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True, output_prob=(args.output_prob_file is not None))
            for worker_id, word_aligns, sent_src, sent_tgt in zip(worker_ids, word_aligns_list, sents_src, sents_tgt):
                output_str = []
                if args.output_prob_file is not None:
                    output_prob_str = []
                if args.output_word_file is not None:
                    output_word_str = []
                for word_align in word_aligns:
                    if word_align[0] != -1:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                        if args.output_prob_file is not None:
                            output_prob_str.append(f'{word_aligns[word_align]}')
                        if args.output_word_file is not None:
                            output_word_str.append(f'{sent_src[word_align[0]]}<sep>{sent_tgt[word_align[1]]}')
                writers[worker_id].write(' '.join(output_str)+'\n')
                if args.output_prob_file is not None:
                    prob_writers[worker_id].write(' '.join(output_prob_str)+'\n')
                if args.output_word_file is not None:
                    word_writers[worker_id].write(' '.join(output_word_str)+'\n')
            tqdm_iterator.update(len(ids_src))

    merge_files(writers)
    if args.output_prob_file is not None:
        merge_files(prob_writers)
    if args.output_word_file is not None:
        merge_files(word_writers)

def return_extended_attention_mask(attention_mask, dtype):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
             "Wrong shape for input_ids or attention_mask"
        )
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

def guess_dtype(model):
  if hasattr(model, 'get_parameter_dtype'):
    return model.get_parameter_dtype()
  elif hasattr(model, 'parameters'):
    return next(model.parameters()).dtype
  else:
    return torch.float32

class ExportNthLayer(torch.nn.Module):
    def __init__(self, base_model, align_layer_max=8):
        super().__init__()
        e = base_model.bert if hasattr(base_model, 'bert') else base_model
        self.bert = e
        self.embeddings = e.embeddings
        # For BERT, num_hidden_layers is in config
        self.config = e.config
        self.num_layers = min(e.config.num_hidden_layers, align_layer_max)
        e = e.encoder if hasattr(e, 'encoder') else e
        self.encoder = e
        self.layer = e.layer[:self.num_layers]
        print(f'{self.layer}')

    def forward(self, ids, attention_mask=None, position_ids=None):
      shape = ids.size()
      device = ids.device
      if attention_mask is None:
        attention_mask = torch.ones(shape, device=device)
        attention_mask[ids==0] = 0
      token_type_ids = torch.zeros(shape, dtype=torch.long, device=device)

      # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
      # ourselves in which case we just need to make it broadcastable to all heads.
      extended_attention_mask = return_extended_attention_mask(attention_mask, guess_dtype(self.bert))

      hidden_states = self.embeddings(ids, token_type_ids=token_type_ids, position_ids=position_ids)

      if self.layer is not None:
        for i, layer in enumerate(self.layer):
          hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)
        return hidden_states
      else:
        return self.bert(ids, attention_mask)

def write_onnx_layers(model, onnx_file_path, max_layer=None,
                      inputs=['input_ids', 'attention_mask'],
                      outputs=['output'],
                      dynamic=True, batch=True, opset_version=14, return_tensor_names=False):
  captions = {0 : 'batch_size', 1: 'sequence_length'} if batch else {0 : 'sequence_length'}
  dynamic_axes = {}
  if dynamic:
    for k in inputs:
      dynamic_axes[k] = captions
    for k in outputs:
      dynamic_axes[k] = captions

  # Create dummy input data
  batch_size = 1
  sequence_length = 128
  dims = (batch_size, sequence_length) if batch else (sequence_length,)
  inputs_ones = tuple(torch.ones(dims) if x != 'input_ids' else torch.randint(0, model.config.vocab_size, dims) for x in inputs)

  # TODO: figure out how to do first nth encoder layers for automodel bert?
  model = ExportNthLayer(model, max_layer) if max_layer is not None else (model.bert if hasattr(model, 'bert') else model)
  torch.onnx.export(
      model,
      inputs_ones, #(input_ids, attention_mask),
      onnx_file_path,
      export_params=True,
      opset_version=opset_version,
      do_constant_folding=True,
      input_names = inputs,
      output_names = outputs,
      dynamic_axes=dynamic_axes,
  )

  if return_tensor_names:
    om = onnx_helper.load_onnx_model(onnx_file_path)
    return list(onnx_helper.enumerate_model_node_outputs(om))
  else:
    return f"Model exported to {onnx_file_path}"

def init_model_and_tokenizer(
    model_name_or_path,
    config_name = None,
    cache_dir = None,
    tokenizer_name = None,
):
  config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
  if config_name:
      config = config_class.from_pretrained(config_name, cache_dir=cache_dir)
  elif model_name_or_path:
      config = config_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
  else:
      config = config_class()

  if tokenizer_name:
      tokenizer = tokenizer_class.from_pretrained(tokenizer_name, cache_dir=cache_dir)
  elif model_name_or_path:
      tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
  else:
      raise ValueError(
          "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
          "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
      )

  # pad is actually always 0
  modeling.PAD_ID = tokenizer.pad_token_id
  modeling.CLS_ID = tokenizer.cls_token_id
  modeling.SEP_ID = tokenizer.sep_token_id

  if model_name_or_path:
      model = model_class.from_pretrained(
          model_name_or_path,
          from_tf=bool(".ckpt" in model_name_or_path),
          config=config,
          cache_dir=cache_dir,
      )
  else:
      model = model_class(config=config)

  return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_file", default=None, type=str, help="The input data file (a text file)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file (required unless output_onnx)."
    )
    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, help='softmax or entmax15'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--output_prob_file", default=None, type=str, help='The output probability file.'
    )
    parser.add_argument(
        "--output_word_file", default=None, type=str, help='The output word file.'
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--output_onnx",
        default=None,
        type=str,
        help="Optionally write onnx conversion of model",
    )
    parser.add_argument(
        "--max_layer",
        type=int,
        default=None,
        help="Limit onnx conversion to this many encoder layers",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    set_seed(args)

    model, tokenizer = init_model_and_tokenizer(args.model_name_or_path, args.config_name, args.cache_dir, args.tokenizer_name)

    if args.output_onnx is not None:
        write_onnx_layers(model, args.output_onnx, max_layer=args.max_layer)

    word_align(args, model, tokenizer)

if __name__ == "__main__":
    main()
