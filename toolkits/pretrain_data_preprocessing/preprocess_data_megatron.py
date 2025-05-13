# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
# 添加上级目录到sys.path以便导入megaton相关模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
#import ftfy  # 可选的文本修复库
import multiprocessing

# 尝试导入NLTK用于句子切分
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

# 引入Megatron中的数据集索引构建工具和tokenizer
from megatron.core.datasets import indexed_dataset
from megatron_patch.tokenizer import build_tokenizer

# 多进程管理器和队列，用于记录总token数量
manager = multiprocessing.Manager()
token_count_queue = multiprocessing.Queue()

# 句子切分占位类，不做实际切分
class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

# 编码器类，处理文本切分和编码
class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.total_token_count = 0  # 用于记录总token数

    # 初始化tokenizer和分句器
    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()

    # 将文本按照最大长度切分并进行句子分割（可选）
    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        total_token_count = 0
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
            total_token_count += sum(len(tokens) for partial in tokens_list for tokens in partial)
        return json.dumps(output), len(json_line), total_token_count

    # 对文本进行编码，转为token id
    def encode(self, json_line):
        try:
            data = json.loads(json_line)
        except:
            return {}, {}, 0, 0
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                if self.args.patch_tokenizer_type in ["DeepSeekV2Tokenizer", "Qwen3Tokenizer", "Qwen2Tokenizer", "LLama3Tokenizer", "LLama2Tokenizer"]:
                    sentence_ids = Encoder.tokenizer.tokenizer(sentence, add_special_tokens=False)['input_ids']
                elif self.args.patch_tokenizer_type == "GPT2BPETokenizer":
                    sentence_ids = Encoder.tokenizer.tokenize(sentence)
                else:
                    sentence_ids = Encoder.tokenizer(sentence, add_special_tokens=False)['input_ids']
                # 跳过空或越界的token
                if not sentence_ids:
                    print(f"tokenizer error sentence_ids is empty :\n {text} \n")
                    continue
                
                if max(sentence_ids) >= Encoder.tokenizer.vocab_size:
                    print(f"tokenizer error max(sentence_ids) >= Encoder.tokenizer.vocab_size :\n {text}\n {max(sentence_ids)}")
                    continue
                
                if len(sentence_ids) > 0:
                    self.total_token_count += len(sentence_ids)
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line), self.total_token_count

# 数据预处理分区逻辑
class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    # 打印处理进度信息
    def print_processing_stats(self, count, proc_start, total_bytes_processed, total_token_count):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s). Total tokens: {total_token_count}.",
                  file=sys.stderr)

    # 对输入文件做句子切分
    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')
        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)
        proc_start = time.time()
        total_bytes_processed = 0
        total_token_count = 0
        for i, (doc, bytes_processed, current_token_count) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            total_token_count += current_token_count
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed, total_token_count)
        fin.close()
        fout.close()

    # 对JSON文件做token编码，写入bin/idx格式
    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        total_token_count = 0  # add token count for process json file
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed, current_token_count) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            total_token_count += current_token_count  # update token count
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed, total_token_count)

        print(f"Total token count: {total_token_count}")  #print total token 
        token_count_queue.put(total_token_count) 

        fin.close()
        builders[key].finalize(output_idx_files[key])



def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=False, default='GPT2BPETokenizer',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'LLama2Tokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    group.add_argument(
        '--patch-tokenizer-type',
        type=str,
        required=True,
        choices=['Qwen3Tokenizer', 'Qwen2Tokenizer', 'LLamaTokenizer', 'DeepSeekV2Tokenizer', 'LLama3Tokenizer', 'LLama2Tokenizer', 'GPT2BPETokenizer'],
        help='What type of tokenizer to use.',
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
                       default=0,
                       help='extra_vocab_size')

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()  # 获取命令行参数配置

    # 如果启用了句子分割，检查是否安装了 nltk 并下载分割器
    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception("nltk library required for sentence splitting is not available.")

    in_ss_out_names = []  # 存储输入、句子分割输出、最终输出路径

    if args.partitions == 1:
        # 单分区处理
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:
        # 多分区时处理多个输入文件
        file_list = os.listdir(args.input)
        in_file_names = [os.path.join(args.input, file) for file in file_list]

        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):  # 统计每个文件的样本行数
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)  # 每个分区的样本数

        # 构造每个分区的输入输出路径
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # 检查分区文件与句子分割文件是否已存在
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        # 若不存在则进行分区
        if not partitions_present and not split_sentences_present:
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names:
                # 支持 gzip 解压
                fin = gzip.open(in_file_name, 'rt') if in_file_name.endswith(".gz") else open(in_file_name, 'r', encoding='utf-8')
                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1) % args.partitions
                fin.close()

            # 关闭所有分区文件
            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    # 确保 worker 数可整除 partitions
    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers // args.partitions)  # 每个分区分配 worker 数

    # 句子分割逻辑（若未完成）
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)
    if args.split_sentences and not split_sentences_present:
        processes = []
        for name in in_ss_out_names:
            # 创建进程分割句子
            p = multiprocessing.Process(target=partition.split_sentences,
                                        args=((name['partition'], name['sentence_split']),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()  # 等待所有子进程完成

        if args.partitions == 1:
            return

    # 编码过程（多进程）
    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # 等待所有进程完成

    if args.partitions == 1:
        return

    level = "sentence" if args.split_sentences else "document"  # 用于命名输出文件

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = build_tokenizer(args)  # 初始化分词器

    # 对每个 json 键分别合并 bin/idx 文件
    for key in args.json_keys:
        output_bin_files[key] = f"{args.output_prefix}_{key}_{level}.bin"
        output_idx_files[key] = f"{args.output_prefix}_{key}_{level}.idx"
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = f"{parition_output_prefix}_{key}_{level}"
            builders[key].add_index(full_partition_output_prefix)

        builders[key].finalize(output_idx_files[key])  # 完成 idx 文件构建

    # 汇总 token 总数
    total_token_count = 0
    while not token_count_queue.empty():
        total_token_count += token_count_queue.get()

    print(f"Total tokens processed: {total_token_count}")

if __name__ == '__main__':

    main()
