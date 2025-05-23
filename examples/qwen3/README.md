# Qwen3 MoE 模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-Core模型训练流程](#Megatron-Core模型训练流程)
      * [模型格式转换](#Megatron-Core模型格式转换)
      * [继续预训练](#预训练示例)
      * [指令微调](#指令微调示例)
   * [下游任务评估](#下游任务评估)
      * [Megatron-Core模型格式转换](#评估格式转换)
      * [运行评估工具](#运行评估工具)


## 安装

请在阿里云人工智能平台PAI产品中填写专属镜像地址： `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04` 

运行下列代码克隆Pai-Megatron-Patch
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
```

目前Qwen3-MoE已支持使用FlashAttention-3加速计算，但只能在Hopper架构的GPU卡上进行运算。若需要在H卡上使用FA3，请在DSW的容器中按如下指令安装并保存镜像
```bash
# 从源码进行安装，需要加上-recurse-submodules以下载相关的子仓库以完成编译
WARNING:DotProductAttention:flash-attn v3 may provide important feature support or performance improvement. Please install flash-attn v3 by
(1) git clone --recurse-modules https://github.com/Dao-AILab/flash-attention.git
(2) cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
(3) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
(4) mkdir -p $python_path/flash_attn_3
(5) wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.pynvidia

# 退出容器后提交更改
docker ps # 查看容器id
docker commit <container-id> qwen3-megatron:25.04 # 提交更改到一个新的镜像

# 为了后续实现模型格式的分布式转换，保存镜像到另外的主机
docker save -o qwen3-megatron.tar qwen3-megatron:25.04
docker load -i qwen3-megatron.tar
```

安装flash-attn3 成功后可以看到
```bash
Using /usr/local/lib/python3.12/dist-packages
Searching for setuptools==75.8.2
Best match: setuptools 75.8.2
Adding setuptools 75.8.2 to easy-install.pth file

Using /usr/local/lib/python3.12/dist-packages
Searching for typing-extensions==4.12.2
Best match: typing-extensions 4.12.2
typing-extensions 4.12.2 is already the active version in easy-install.pth

Using /usr/local/lib/python3.12/dist-packages/setuptools/_vendor
Searching for filelock==3.17.0
Best match: filelock 3.17.0
Adding filelock 3.17.0 to easy-install.pth file

Using /usr/local/lib/python3.12/dist-packages
Searching for MarkupSafe==3.0.2
Best match: MarkupSafe 3.0.2
Adding MarkupSafe 3.0.2 to easy-install.pth file

Using /usr/local/lib/python3.12/dist-packages
Searching for mpmath==1.3.0
Best match: mpmath 1.3.0
Adding mpmath 1.3.0 to easy-install.pth file

Using /usr/local/lib/python3.12/dist-packages
Finished processing dependencies for flash-attn-3==3.0.0b1
```

## 预训练数据集和模型下载

```bash
cd /mnt
mkdir qwen-ckpts
cd qwen-ckpts
git clone https://www.modelscope.cn/Qwen/Qwen3-30B-A3B.git

mkdir qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-train-general.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-valid-general.json


```

## Megatron-Core模型训练流程
### Megatron-Core模型格式转换
当前qwen3已升级至`torch_dist`格式权重训练，为了进行权重转换，需要传入的参数列表如下
```
MODEL_SIZE=$1               # 模型大小，0.6B, 1.7B, 4B, 8B, 14B, 32B, A3B, A22B
LOAD_DIR=$2                 # 源权重路径
SAVE_DIR=$3                 # 目标权重路径
MG2HF=$4                    # 转换方向 可选: true, false
USE_CUDA=$5                 # 是否使用GPU转换 建议: true
PR=$6                       # 转换精度 可选: fp32 bf16 fp16
HF_DIR=$7                   # HF权重路径(mcore2hf时必须提供)
```
例如，使用下述脚本将checkpoint转换到MCore格式

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3/run_8xH20.sh \
A3B \
/mnt/qwen-ckpts/Qwen3-30B-A3B \
/mnt/qwen-ckpts/Qwen3-30B-A3B-to-mcore  \
false \
true \
bf16
```

如果需要自定义转换脚本，请参阅分布式转换工具。


**修改scripts/qwen3/run_8xH20.sh 中的TP，PP，EP参数从命令行中传入**

```bash
# vim scripts/qwen3/run_8xH20.sh


TP=$7
PP=$8
EP=$9
HF_DIR=${10} 

elif [ $MODEL_SIZE = A3B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 48
        --hidden-size 2048
        --ffn-hidden-size 6144
        --moe-ffn-hidden-size 768
        --num-attention-heads 32
        --untie-embeddings-and-output-weights
        --moe-grouped-gemm
        --moe-router-score-function softmax
        --moe-token-dispatcher-type alltoall
        --moe-router-topk 8
        --moe-layer-freq "'([1]*48)'"
        --num-experts 128
        --num-query-groups 4
    )
    if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size ${TP}
            --pipeline-model-parallel-size ${PP}
            --expert-model-parallel-size ${EP}
        )
    fi
```
```bash
MODEL_SIZE=$1               # 模型大小，0.6B, 1.7B, 4B, 8B, 14B, 32B, A3B, A22B
LOAD_DIR=$2                 # 源权重路径
SAVE_DIR=$3                 # 目标权重路径
MG2HF=$4                    # 转换方向 可选: true, false
USE_CUDA=$5                 # 是否使用GPU转换 建议: true
PR=$6                       # 转换精度 可选: fp32 bf16 fp16
TP=$7
PP=$8
EP=$9
HF_DIR=${10}                   # HF权重路径(mcore2hf时必须提供)
```
```bash
cd /xxx/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/
bash scripts/qwen3/run_8xH20.sh \
A3B \
/xxx/Qwen3-30B-A3B/Qwen3-30B-A3B \
/xxx/qwen-ckpts/Qwen3-30B-A3B-to-mcore  \
false \
true \
bf16 \
4 \
2 \
2 \
```


### Megatron-Core预训练及指令微调
在Qwen3 MoE中，我们已将预训练和微调整合到`run_mcore_qwen3.sh`脚本，对于不同的使用场景，二者各参数的意义有所不同。

#### 预训练&微调命令统一描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 0.6B, 1.7B, 4B, 8B, 14B, 32B, A3B, A22B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度
PR=${9}                         # 训练精度: fp16, bf16, fp8
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
CP=${12}                        # 上下文并行度
ETP=${13}                       # 专家张量并行度
EP=${14}                        # 专家模型并行度
SP=${15}                        # 是否使用序列并行: true, false
DO=${16}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${17}                        # 是否优先使用Flash Attention: true, false
SFT=${18}                       # 是否执行微调训练: true, false
AC=${19}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${20}         # 是否启用Offload optimizer: false, 或输入0～1的小数作为参数offload比例
SAVE_INTERVAL=${21}             # 保存ckpt的间隔
DATASET_PATH=${22}              # 训练数据集路径
VALID_DATASET_PATH=${23}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${24}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${25}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${26}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${27}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对qwen2的继续预训练。
备注：当`AC=offload`或`full`时，可设置`MP_AC_LAYERS`环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：`1`）。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen3
sh run_mcore_qwen3.sh  \
dlc  \
A3B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
4   \
2  \
1 \
1 \
4 \
true \
true   \
true \
false \
sel   \
false \
100000  \
/mnt/qwen-datasets/mmap_qwen3_datasets_text_document   \
/mnt/qwen-datasets/mmap_qwen3_datasets_text_document   \
/mnt/qwen-ckpts/Qwen3-30B-A3B-to-mcore  \
10000  \
100   \
/mnt/logs/output_mcore_qwen3_pretrain
```

运行预训练脚本报错，
```bash
[rank9]: Traceback (most recent call last):
[rank9] :
[rank9]: File "/data/tujie/Qwen3/Pai-Megatron-Patch/examples/qwen3/pretrain_qwen.py", line 141, in < module>
pretrain(
[rank9] :
File "/data/tujie/Qwen3/Pai-Megatron-Patch/Megatron-LM-250328/megatron/training/training.py", line 726, in pretrain
[rank9] :
iteration, num_floating_point_operations_so_far = train(
[rank9] :
AAAAAA
[rank9] :
File "/data/tujie/Owen3/Pai-Megatron-Patch/Megatron-LM-250328/megatron/training/training.py", line 1909, in train
[rank9] :
train_step(forward_step_func,
[rank9] :
File "/data/tujie/Qwen3/Pai-Megatron-Patch/Megatron-LM-250328/megatron/training/training.py", line 1172, in train_step
[rank9]:
losses_reduced = forward_backward_func(
[rank9] :
AAAAAAAAAAAAAAAAAAAAAA

[rank9]: File "/data/tujie/Qwen3/Pai-Megatron-Patch/Megatron-LM-250328/megatron/core/pipeline_parallel/schedules.py", line 1918, in

forward_backward_pipelining_without_interleaving
[rank9] :
input_tensor-grad = backward_step(
[rank9] :
ЛААААААААААААА
[rank9] :
File "/data/tujie/Qwen3/Pai-Megatron-Patch/Megatron-LM-250328/megatron/core/pipeline_parallel/schedules.py"
, line 390, in backward_step
Trank9
custom_backwardoutput_tensor [0], output_tensor-grad[0])
[rank9] :
File "/data/tujie/Qwen3/Pai-Megatron-Patch/Megatron-LM-250328/megatron/core/pipeline_parallel/schedules.py"
, line 151, in custom_backward
[rank9]:
Variable._execution_engine.run_backward(
frank9] :
File "/usr/local/lib/python3.12/dist-packages/torch/autograd/function.py", line 307, in apply
• ink9] :
return user_fn(self, *args)
[rank9]:
AAAAAAAAAAAAAAAAAAAA
[rank9] :
File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/grouped_linear.py", line 270, in backward
[rank9] :
general_grouped_gemm(
Trank9:
File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/cpp_extensions/gemm.py", line 206, in general_grouped_gemm
[rank9]:
bias = tex.te_general_grouped_gemm(l
Trank9:
AAAAAAAAAAAAAAAAAAAAAAAAAAAA

[rank9]: RuntimeError: The specified pointer resides on host memory and is not registered with any CUDA device.
```
这是moe grouped gemm 的 bug，更新TE 到v2.2版本可解决问题
```bash
vim /etc/pip/constraints.txt # 删除transformer-engine那行
git clone --recurse-submodules https://github.com/NVIDIA/TransformerEngine.git

cd TransformerEngine
git tag
git checkout v2.2.1 # 2.2版本安装不成功，TE和PyTorch无法建立连接
export NVTE_FRAMEWORK=pytorch   # Optionally set framework
pip3 install . -vvv # Build and install
```

#### 指令微调示例
制作idxmap用于微调的数据集可以参考[链接](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing)。
当准备好微调数据集后，将SFT开关设置为`true`即可进行指令微调。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen3
sh run_mcore_qwen3.sh  \
dlc  \
A3B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
4   \
2  \
1 \
1 \
4 \
true \
true   \
true \
true \
sel   \
false \
100000  \
/mnt/qwen-datasets/path_to_your_dataset   \
/mnt/qwen-datasets/path_to_your_dataset   \
/path/to/pretraining/checkpoint  \
10000  \
100   \
/workspace/output_mcore_qwen3_finetune
```
通过设置MP_DATASET_TYPE环境变量，本脚本还可使用json格式的数据集进行指令微调
```bash
export MP_DATASET_TYPE="raw"
cd /workspace/Pai-Megatron-Patch/examples/qwen3
sh run_mcore_qwen3_moe.sh  \
dlc  \
A3B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
4   \
2  \
1 \
1 \
4 \
true \
true   \
true \
true \
sel   \
false \
100000  \
/mnt/qwen-datasets/alpaca_zh-train-general.json    \
/mnt/qwen-datasets/alpaca_zh-valid-general.json   \
/mnt/qwen-ckpts/Qwen3-30B-A3B-to-mcore  \
10000  \
100   \
/workspace/output_mcore_qwen3_finetune
```

## 下游任务评估

### 评估格式转换
您需要将训练/微调后保存的Megatron-Core转换为HuggingFace格式来进行推理评估。

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3/run_8xH20.sh \
A3B \
/mnt/qwen-ckpts/Qwen3-30B-A3B-to-mcore \
/mnt/qwen-ckpts/Qwen3-30B-A3B-mcore-to-hf  \
true \
true \
bf16 \
/mnt/qwen-ckpts/Qwen3-30B-A3B
```

### 运行评估工具
下载评估数据
```bash
# In container
cd /workspace

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/evaluate.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/cmmlu.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/ceval.tgz 

tar -xvzf cmmlu.tgz 
tar -xvzf ceval.tgz 
tar -xvzf evaluate.tgz
```
运行以下指令对转换后的模型进行评估。
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/qwen-ckpts/Qwen3-30B-A3B-mcore-te-to-hf,trust_remote_code=True \
--tasks cmmlu,ceval-valid  \
--batch_size 16
```
