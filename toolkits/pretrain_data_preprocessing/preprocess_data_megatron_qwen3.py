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
from datasets import load_dataset
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
total_sample_count_queue = multiprocessing.Queue()
total_attn_mask_zero_queue = multiprocessing.Queue()

# 句子切分占位类，不做实际切分
class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

# 编码器类，处理文本切分和编码
class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.total_token_count = 0  # 用于记录总token数
        self.total_attn_mask_zero = 0     # 用于记录attention_mask含0的样本个数
        self.total_samples = 0      # 记录样本个数

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
    # 对于我们的arrow文件，已经切分好了，arrow文件中的每条数据都是8192的长度
    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        total_token_count = 0
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            # 使用滑动窗口进行数据集的制作，这是nanogpt的做法，适用于数据集比较小时
            # 现有业务数据量已经达到这个水平，因此，不再需要sliding windows
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
            total_token_count += sum(len(tokens) for partial in tokens_list for tokens in partial)
        return json.dumps(output), len(json_line), total_token_count
    
    def process_arrow(self,arrow_line):
        #[TODO]
        # print(arrow_line)
        # {'input_ids': [99, 75847, 31877, 109434, 42414, ...], 'attention_mask': [1,...]}
        ids={}
        lens={}
        for key in self.args.json_keys:
            wheather_attn_mask_zero = False # 判断当前数据的attention_mask是否含零
            input_ids = arrow_line["input_ids"]
            attention_mask = arrow_line["attention_mask"]
            
            if len(input_ids) > 0 and self.args.append_eod:
                input_ids[-1] = Encoder.tokenizer.eod
            
            sequence_len = sum(attention_mask)
            self.total_token_count += sequence_len
            if len(input_ids) != sequence_len:# 如果attention_mask不全为1，那么取出有效长度部分的input_ids
                self.total_attn_mask_zero += 1
                input_ids = input_ids[:sequence_len]
            ids[key] = input_ids
            lens[key] = [sequence_len]
            self.total_samples += 1
        return ids,lens,sequence_len,self.total_token_count,self.total_attn_mask_zero,self.total_samples
        
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
        # 多进程处理 fin 生成的文档，调用 encoder.encode 进行编码。
        # fin 是一个文档生成器，代表了所有输入文件中按顺序读取的数据流，用于供多进程进行文本编码处理。
        # fin每次返回一个jsonl文档，包含了一行json数据
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        # pool.imap 是 Python 多进程库 multiprocessing 中的一个方法，用于并发地按顺序处理可迭代对象，是对 map 的惰性（lazy）版本。 （惰性迭代不会一次将所有数据放到内存中）
        # 使用多进程池，把 fin 生成的一条条 JSON 文档，分批（每 32 条）并行送到 encoder.encode() 中处理，按顺序返回处理结果。
        encoded_docs = pool.imap(encoder.encode, fin, 32)
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        # {output_prefix}_{key}_document.bin：token ID 的二进制数据
        # {output_prefix}_{key}_document.idx：索引文件，标识每条记录的位置
        for key in self.args.json_keys:
            # 针对每个指定的 jsonl_key（例如 'text'），构建一个 .bin 和 .idx 文件
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
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
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        total_token_count = 0  # add token count for process json file
        print("Time to startup:", startup_end - startup_start)
        # 每个 doc 是由 encoder 处理过的 tokenized 文本。
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
        
    def process_arrow_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        
        # fin = open(input_file_name, 'r', encoding='utf-8')
        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        # 多进程处理 fin 生成的文档，调用 encoder.encode 进行编码。
        # fin 是一个文档生成器，代表了所有输入文件中按顺序读取的数据流，用于供多进程进行文本编码处理。
        # fin每次返回一个jsonl文档，包含了一行json数据
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        # pool.imap 是 Python 多进程库 multiprocessing 中的一个方法，用于并发地按顺序处理可迭代对象，是对 map 的惰性（lazy）版本。 （惰性迭代不会一次将所有数据放到内存中）
        # 使用多进程池，把 fin 生成的一条条 JSON 文档，分批（每 32 条）并行送到 encoder.encode() 中处理，按顺序返回处理结果。
        # encoded_docs = pool.imap(encoder.encode, fin, 32)
        dataset = load_dataset("arrow",data_files=input_file_name)
        data=dataset["train"]
        encoded_docs = pool.imap(encoder.process_arrow, data, 32)
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        # {output_prefix}_{key}_document.bin：token ID 的二进制数据
        # {output_prefix}_{key}_document.idx：索引文件，标识每条记录的位置
        for key in self.args.json_keys:
            # 针对每个指定的 jsonl_key（例如 'text'），构建一个 .bin 和 .idx 文件
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
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
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        total_token_count = 0  # add token count for process json file
        total_samples_count = 0
        total_attn_mask_zero_count = 0
        print("Time to startup:", startup_end - startup_start)
        # 每个 doc 是由 encoder 处理过的 tokenized 文本。
        for i, (doc, sentence_lens, bytes_processed, current_token_count,current_attn_mask_zero,current_sampels) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            total_token_count += current_token_count  # update token count
            total_samples_count += current_sampels
            total_attn_mask_zero_count += current_attn_mask_zero
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed, total_token_count)

        print(f"Total token count: {total_token_count}")  #print total token 
        print(f"Total sample count:{total_samples_count}")
        print(f"Total attn mask zero count:{total_attn_mask_zero_count}")
        token_count_queue.put(total_token_count) 
        total_sample_count_queue.put(total_samples_count)
        total_attn_mask_zero_queue.put(total_attn_mask_zero_count)
        
        # fin.close()
        builders[key].finalize(output_idx_files[key])

    

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    """
    使用 action='store_true'，默认情况下：
        - 不传递参数时 → 值为 False
        - 传递参数时（如 --load_from_arrow）→ 值为 True
    """
    group.add_argument('--load_from_arrow',action='store_true',help="whether load from arrow file")
    # argparse 的 add_argument 默认 required=False，所以可以去掉。
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
    # 将 args.input 的文件路径（如 "data/input.txt"）按扩展名分割：
    # file_name = "data/input"
    # extension = ".txt"
    file_name, extension = os.path.splitext(args.input)
    # 构造带编号的输入文件名，如："data/input_0.txt"
    input_file_name = file_name + "_" + str(file_id) + extension
    # 构造带 _ss_编号 的句子切分文件名，如："data/input_ss_0.txt"
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    """
    'partition': 分片后的输入文件名
    'sentence_split': 句子切分后的文件名
    'output_prefix': 输出前缀（用于生成多个输出文件）
    """
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

    if args.partitions == 1: # 传入的是单个文件
        # 单分区处理
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:   # 传入的args.input是多个文件
        # 多分区时处理多个输入文件
        file_list = os.listdir(args.input)
        in_file_names = [os.path.join(args.input, file) for file in file_list]

        # 读取所有的文件然后根据partision_size重新组织输入文件
        # 也就是说每个文件中的行数被固定为partision_size
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    # fc从0开始索引
                    for fc, _ in enumerate(fin):  # 统计每个文件的样本行数
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)  # 每个分区的样本数

        # 构造每个分区的输入输出路径
        """
        'partition': 分片后的输入文件名
        'sentence_split': 句子切分后的文件名
        'output_prefix': 输出前缀（用于生成多个输出文件）
        file_names = {
            'partition': input_file_name,
            'sentence_split': sentence_split_file,
            'output_prefix': output_prefix}
        return file_names
        """
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
                if not args.load_from_arrow:
                    fin = gzip.open(in_file_name, 'rt') if in_file_name.endswith(".gz") else open(in_file_name, 'r', encoding='utf-8')
                else:
                    # [TODO] 在这里增加arrow文件的读取
                    pass
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
    # 在这里进行句子切分，按照max_len来进行分割
    # args.split_sentences默认值为False
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
        if not args.load_from_arrow:
            p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        else:
            p = multiprocessing.Process(target=partition.process_arrow_file,
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
    total_sample_count = 0
    total_zero_mask_count = 0
    while not token_count_queue.empty():
        total_token_count += token_count_queue.get()
    while not total_sample_count_queue.empty():
        total_sample_count += total_sample_count_queue.get()
    while not total_attn_mask_zero_queue.empty():
        total_zero_mask_count += total_attn_mask_zero_queue.get()
        

    print(f"Total tokens processed: {total_token_count}\n Total samples processed: {total_sample_count}\n Total zero attn mask processed: {total_zero_mask_count}")

if __name__ == '__main__':

    main()
