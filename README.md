# Introduction

TinyLLM是我学习大语言模型的一个小项目，目标是尝试从零开始训练一个非常轻量的语言模型。

项目目前包含：

- 一个简单的模型实现（包括基础的Dense和MoE结构）
- 基本的预训练和指令微调流程
- 使用了一些常见的开源框架，如`transformers`, `accelerate`, `peft`等
- 支持在单张和多张GPU上训练
- 包含了一些简单的模型测试代码

这个项目记录了我的学习过程，肯定还有很多不足和需要改进的地方。希望能和大家一起交流学习，共同进步！

# Environment

仅是我个人的软硬件环境配置，自行酌情更改：

* Ubuntu == 22.04
* Python == 3.10
* Pytorch == 2.1.2
* CUDA == 12.1
* [requirements.txt](./requirements.txt)

# Quick Inference & Test

</div>

```bash
# step 1
git clone https://github.com/VocabVictor/tinyllm.git
# or 
git clone https://gh-proxy.com/https://github.com/VocabVictor/tinyllm.git
```

```bash
# step 2
python eval/eval.py
```

或者启动streamlit，启动网页聊天界面

```bash
# or step 3, use streamlit
streamlit run inference/fast_inference.py
```

# Quick Start

* 0、环境配置
  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
* 1、获取项目代码
  ```bash
  git clone https://github.com/VocabVictor/tinyllm.git
  # or
  git clone https://gh-proxy.com/https://github.com/VocabVictor/tinyllm.git
  ```
* 2、如需自行训练模型

    * 2.1 从[指定数据集地址](#数据集下载地址)获取数据集并置于`./dataset`目录

    * 2.2 执行`python data_process.py`对数据集进行处理，例如对预训练数据进行token编码、从SFT数据集中提取问答对至CSV文件

    * 2.3 在`./model/LMConfig.py`中调整模型参数配置
    * 2.4 执行`python pretrain/pretrain.py`进行预训练
    * 2.5 执行`python sft/full_sft.py`进行指令微调
    * 2.6 执行`python sft/lora_sft.py`进行LoRA微调（可选）
    * 2.7 执行`python rlhf/dpo_train.py`进行基于人类偏好的强化学习对齐（可选）

* 3、验证模型推理效果
    * 确保训练完成的模型权重文件位于`./out/`目录
       ```text
      out
      ├── multi_chat
      │   ├── full_sft_512.pth
      │   ├── full_sft_512_moe.pth
      │   └── full_sft_768.pth
      ├── single_chat
      │   ├── full_sft_512.pth
      │   ├── full_sft_512_moe.pth
      │   └── full_sft_768.pth
      ├── pretrain_768.pth
      ├── pretrain_512_moe.pth
      ├── pretrain_512.pth
      ```
    * 执行`python eval/eval_pretrain.py`测试预训练模型的文本生成能力
    * 执行`python eval/eval.py`测试模型的对话能力

预训练和全参微调过程（包括pretrain和full_sft）均支持多GPU并行加速训练。以下是在单机多GPU环境下启动训练的方法：

* 使用分布式数据并行(DDP)进行训练：
  ```bash
  torchrun --nproc_per_node N pretrain/pretrain.py
  torchrun --nproc_per_node N sft/full_sft.py
  ```

* 使用DeepSpeed框架进行训练：
  ```bash
  deepspeed --master_port 29500 --num_gpus=N pretrain/pretrain.py
  deepspeed --master_port 29500 --num_gpus=N sft/full_sft.py
  ```

注：在上述命令中，N代表要使用的GPU数量。请根据实际硬件配置调整该参数。

* 记录训练过程
    ```bash
    torchrun --nproc_per_node N pretrain/pretrain.py --use_wandb
    # and
    torchrun --nproc_per_node N sft/full_sft.py --use_wandb
    ```
    通过添加`--use_wandb`参数，可以记录训练过程，训练完成后，可以在wandb网站上查看训练过程。通过修改`wandb_project`和`wandb_run_name`参数，可以指定项目名称和运行名称。

# Data sources

- 🤖 分词器：分词器就像一本特殊的词典，它把单词转换成数字。这样做是为了让计算机更容易理解和处理文字。

  有两种方法可以得到分词器：
  1. 自己创建一个：可以用`pretrain/train_tokenizer.py`来做这件事。
  2. 使用现成的：直接用其他人已经做好的分词器。

  分词器的词表（就是这本"词典"里有多少个词）大小很重要：
  - 太大：模型会变得很大，可能会影响运行速度。
  - 太小：可能无法准确表达所有需要的词。

  下面是一些知名模型使用的分词器及其词表大小：

<table>
  <tr><th>分词器名称</th><th>词表大小</th><th>来自哪里</th></tr>
  <tr><td>yi tokenizer</td><td>64,000</td><td>01万物（中国）</td></tr>
  <tr><td>qwen2 tokenizer</td><td>151,643</td><td>阿里云（中国）</td></tr>
  <tr><td>glm tokenizer</td><td>151,329</td><td>智谱AI（中国）</td></tr>
  <tr><td>mistral tokenizer</td><td>32,000</td><td>Mistral AI（法国）</td></tr>
  <tr><td>llama3 tokenizer</td><td>128,000</td><td>Meta（美国）</td></tr>
</table>

  本项目使用了一个相对较小的词表，这样可以让模型更小巧，更容易在普通电脑上运行。

### 数据集下载地址

下载到`./dataset/`目录下

| 数据集      | 下载地址                                                                                                                                                                                                                       |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **【tokenizer训练集】** | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) / [百度网盘](https://pan.baidu.com/s/1yAw1LVTftuhQGAC1Y9RdYQ?pwd=6666)                                                                   |
| **【Pretrain数据】**   | [Seq-Monkey官方](http://share.mobvoi.com:5000/sharing/O91blwPkY)  / [百度网盘](https://pan.baidu.com/s/1-Z8Q37lJD4tOKhyBs1D_6Q?pwd=6666) / [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) |
| **【SFT数据】**        | [匠数大模型SFT数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl)                                                                                                              |
| **【DPO数据】**        | [Huggingface](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main/dpo)                                                                                                                                  |

# Model

Dense模型架构（与[Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)相似）采用了Transformer的Decoder-Only结构，相较于GPT-3有以下主要区别：

* 实现了Pre-Normalization：在每个Transformer子层的输入端应用归一化，而非输出端。具体使用RMSNorm（Root Mean Square Layer Normalization）作为归一化函数。
* 激活函数选择：用SwiGLU（Swish-Gated Linear Unit）替换了ReLU，以提升模型性能。
* 位置编码方案：摒弃了绝对位置嵌入，采用旋转位置编码（Rotary Position Embedding, RoPE），这种方法在处理超出训练序列长度的推理任务时表现更为出色。

MoE（Mixture of Experts）模型架构基于[Deepseek-V2](https://arxiv.org/pdf/2405.04434)中提出的MixFFN（Mixed Feed-Forward Network）混合专家模块：

* DeepSeek-V2在前馈网络（FFN）层面引入了更细粒度的专家分割策略和共享的专家隔离技术，旨在提升各个Expert的效能和整体模型性能。

---

# Experiment

```bash
CPU: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
内存：256 GB
显卡：NVIDIA Tesla V100(32GB) * 4
环境：python 3.9 + Torch 2.1.2 + DDP多卡训练
```

---

1. **预训练(Text-to-Text)**:
    - 预训练阶段模型从大量文本数据中学习基础知识和语言模式。
    - 数据来源包括维基百科、新闻、书籍等广泛领域。
    - 目标是通过无监督学习压缩知识到模型权重，提高下一词预测能力。

    ```bash
    torchrun --nproc_per_node 2 pretrain/pretrain.py
    ```
2. **单轮对话有监督微调**:
    - 对预训练模型进行微调，使其适应聊天模板。
    - 限制模型在指定格式内生成回复。
    - 指令和回答长度限制为512 tokens，以优化资源使用。

   ```bash
   torchrun --nproc_per_node 2 sft/full_sft.py
   ```

3. **多轮对话微调**:
    - 在单轮对话基础上，进行多轮对话模板的微调。
    - 使用数据集中的历史对话和回答字段构建训练样本。
    - 注意：小模型在长上下文对话中表现可能不佳。

    ```bash
    torchrun --nproc_per_node 2 sft/full_sft.py
    ```

4. **直接偏好优化（DPO）**:
    - 通过强化学习方法，优化模型输出以更符合人类偏好。
    - 使用正面和负面示例进行训练。

    ```bash
    python rlhf/dpo_train.py
    ```

---

# Others

### 推理与导出

* [export/export_model.py](export/export_model.py)可以导出模型到transformers格式，推送到huggingface

---

### API推理

* [inference/my_openai_api.py](inference/my_openai_api.py)完成了openai_api的聊天接口，方便将自己的模型接入第三方UI
  例如fastgpt、OpenWebUI等

* 启动聊天服务端
    ```bash
    python inference/my_openai_api.py
    ```
* 测试服务接口
    ```bash
    python inference/chat_openai_api.py
    ```
* API接口示例，兼容openai api格式
    ```bash
    curl http://ip:port/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{ 
        "model": "model-identifier",
        "messages": [ 
          { "role": "user", "content": "世界上最高的山是什么？" }
        ], 
        "temperature": 0.7, 
        "max_tokens": -1,
        "stream": true
    }'
    ```

# Acknowledge

<details close> 
<summary> <b>感谢以下优秀的开源项目</b> </summary>

- 排名不分任何先后顺序
- [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
- [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- [https://github.com/DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
- [https://github.com/charent/ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
- [https://github.com/wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh)
- [https://github.com/Tongjilibo/build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
- [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- [https://github.com/AI-Study-Han/Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt)
- [https://github.com/xusenlinzy/api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [https://github.com/HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)

</details>


# License

This repository is licensed under the [Apache-2.0 License](LICENSE).


