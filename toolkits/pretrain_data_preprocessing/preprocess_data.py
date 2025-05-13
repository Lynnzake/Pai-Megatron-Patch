# Copyright (c) 2023 Alibaba PAI Team.  All rights reserved.
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
"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys
import time
from threading import Semaphore
import torch
import ftfy
import lm_dataformat as lmd
import tqdm

from megatron.core.datasets import indexed_dataset
from megatron_patch.tokenizer import build_tokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            try:
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='do_not_pad',max_length=32768,truncation=True)['input_ids']
                """
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='max_length',
                                             max_length=2047, truncation=True)['input_ids']
                """
                if max(text_ids) >= Encoder.tokenizer.vocab_size:
                    print(text)
                    print(max(text_ids))
                    continue
            except Exception as e:
                print(f"Error encoding text: {e}")  # print error message
                continue
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                if hasattr(Encoder.tokenizer, 'eos_token_id'):
                    doc_ids[-1].append(Encoder.tokenizer.eos_token_id)
                elif hasattr(Encoder.tokenizer, 'eod_id'):
                    doc_ids[-1].append(Encoder.tokenizer.eod_id)
                else:
                    doc_ids[-1].append(Encoder.tokenizer.eod)
                #doc_ids[-1].append(Encoder.tokenizer.pad_token_id)
            ids[key] = doc_ids
        return ids, len(text)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True)
    group.add_argument(
        '--jsonl-keys',
        nargs='+',
        default=['content'],
        help='space separate listed of keys to extract from jsonl. Defa',
    )
    group.add_argument(
        '--num-docs',
        default=None,
        type=int,
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--patch-tokenizer-type',
        type=str,
        required=True,
        choices=[
            'JiebaBPETokenizer', 'BloomTokenizerFromHF',
            'ChatGLMTokenizerFromHF', 'GPT2BPETokenizer',
            'GLM10BZHTokenizerFromHF', 'IcetkGLM130BTokenizer',
            'LLamaTokenizer', 'FalconTokenizer', 'OPTTokenizer',
            'StarcoderTokenizerFromHF', 'QwenTokenizer','Qwen2Tokenizer', 'MistralTokenizer'
        ],
        help='What type of tokenizer to use.',
    )
    group.add_argument('--vocab-file',
                       type=str,
                       default=None,
                       help='Path to the vocab file')

    group.add_argument(
        '--merge-file',
        type=str,
        default=None,
        help='Path to the BPE merge file (if necessary).',
    )
    group.add_argument(
        '--append-eod',
        action='store_true',
        help='Append an <eod> token to the end of a document.',
    )
    group.add_argument('--ftfy',
                       action='store_true',
                       help='Use ftfy to clean text')
    group = parser.add_argument_group(title='output data')
    group.add_argument(
        '--output-prefix',
        type=str,
        required=True,
        help='Path to binary output file without suffix',
    )
    group.add_argument(
        '--dataset-impl',
        type=str,
        default='mmap',
        choices=['lazy', 'cached', 'mmap'],
        help='Dataset implementation to use. Default: mmap',
    )

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers',
                       type=int,
                       default=1,
                       help='Number of worker processes to launch')
    group.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='Interval between progress updates',
    )
    group.add_argument('--load',
                       type=str,
                       default=None,
                       help='path to tokenizer config file')
    group.add_argument('--seq-length',
                       type=int,
                       default=2048,
                       help='sequence length')
    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=1,
                       help='extra_vocab_size')
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


"""
信号量（Semaphore）用于限制同时处理的文档数量。在主处理逻辑中我们看到：

semaphore = Semaphore(10000 + args.workers)
这意味着最多允许 10000 + workers 个数据项被处理或挂起在内存中，防止爆内存或过度打开文件。

处理完每条数据后，会手动 semaphore.release()，释放信号量：

for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
    ...
    semaphore.release()
"""
def yield_from_files(fnames: list, semaphore):
    """
    - fnames: 一个文件路径列表。
    - semaphore: 一个多线程/多进程中的信号量，用于限制并发数（例如限制内存使用或 I/O 访问）。
    """
    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            # lmd.Reader(fname).stream_data()：逐条读取文件中数据（通常是 JSONL）。
            # filter(lambda x: x, ...)：过滤掉空行或无效数据。
            # 每读取一条数据，就先 semaphore.acquire()，表示占用一个处理资源位。
            # 然后将这条数据 yield 出去。
            semaphore.acquire()
            yield f
    # 主循环：
    # 对每个文件：
    # 先占用一个信号量（代表要打开并处理这个文件）。
    # 然后用 yield from 把 yielder(fname) 生成的每条数据逐条输出。
    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def main():
    args = get_args()
    args.tensor_model_parallel_size = 1
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.vocab_extra_ids = 0
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f'Vocab size: {tokenizer.vocab_size}')
    print(f'Output prefix: {args.output_prefix}')

    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    # 遍历输入目录 args.input 下的所有文件
    # yield_from_files 是一个生成器，逐行从文件中读取数据（并受 Semaphore 控制，以限制内存使用）
    file_list = os.listdir(args.input)
    path_list = [os.path.join(args.input, file) for file in file_list]
    fin = yield_from_files(path_list, semaphore)

    if args.workers > 1:
        # 多进程处理 fin 生成的文档，调用 encoder.encode 进行编码。
        # fin 是一个文档生成器，代表了所有输入文件中按顺序读取的数据流，用于供多进程进行文本编码处理。
        # fin每次返回一个jsonl文档，包含了一行json数据
        pool = multiprocessing.Pool(args.workers,
                                    initializer=encoder.initializer)
        # 编码结果应为字典形式，如：{"text": [[1, 2, 3], [4, 5, 6]]}
        # pool.imap 是 Python 多进程库 multiprocessing 中的一个方法，用于并发地按顺序处理可迭代对象，是对 map 的惰性（lazy）版本。 （惰性迭代不会一次将所有数据放到内存中）
        # 使用多进程池，把 fin 生成的一条条 JSON 文档，分批（每 25 条）并行送到 encoder.encode() 中处理，按顺序返回处理结果。
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    # {output_prefix}_{key}_document.bin：token ID 的二进制数据
    # {output_prefix}_{key}_document.idx：索引文件，标识每条记录的位置
    for key in args.jsonl_keys:
        # 针对每个指定的 jsonl_key（例如 'text'），构建一个 .bin 和 .idx 文件
        output_bin_files[key] = '{}_{}_{}.bin'.format(args.output_prefix, key,
                                                      'document')
        output_idx_files[key] = '{}_{}_{}.idx'.format(args.output_prefix, key,
                                                      'document')
        # 使用 IndexedDatasetBuilder 创建数据集构建器（写入二进制文件用）
        """
        indexed_dataset.IndexedDatasetBuilder(...) 是用于构建 高效可索引的数据集文件 的类，输出两个文件：
            - .bin 文件：保存所有 token ID，连续的二进制数据
            - .idx 文件：保存每一条记录的起止位置，支持快速随机访问
        这个结构被 Megatron-LM、Fairseq 等用于加载预训练数据。
        """
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    # 每个 doc 是由 encoder 处理过的 tokenized 文本。
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        semaphore.release()

        # add each tokenized document / sentence
        # 每个 sentence 是一个 token_id 序列（即一个段落或句子）。
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            # separate with eos token
            # .end_document() 会在文档之间插入特殊分隔（如 <eos> token）。
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(f'Processed {i} documents '
                                 f' ({i / elapsed} docs/s, {mbs} MB/s).')
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        # .finalize() 会写入 .idx 文件并关闭 .bin 文件
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()
