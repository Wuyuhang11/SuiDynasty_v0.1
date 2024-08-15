### 隋朝那些事儿的项目文档

#### 1.项目介绍

**项目背景和动机：** 隋朝在中国历史上是一个历史短暂的国家，但它的出现和贡献却不容忽视。隋朝在中国历史上起到了承上启下的重要作用，特别是在政治、经济和文化方面的改革与创新，对后世产生了深远影响。但是隋朝的相关数据，尤其诗词方面在现在社会上却少有流传，但是不可否认的是，隋朝诗词在中国文化起到了不可或缺的作用，尤其是隋文帝时期编纂的《隋书》是中国历史上重要的史书之一。而目前的大模型（GPT-4、Qwen等）对于隋朝的知识较为匮乏，因此我们利用`InternLM`提供的`1.8B`大模型在隋朝诗词数据上进行训练，并在模型`QA`的上游部分整合了内外部知识进行了知识填充。以较低的训练成本得到一个较高的响应内容。

**Base模型来源：** 本项目基于`InternLM`大模型进行训练，模型base参考：https://github.com/InternLM/Tutorial

**数据来源：** 本项目数据`90%`参考https://github.com/CanvaChen/llm-dataset-chinese-poetry/blob/main/data2 ，共1w条左右

**主要工作：** 我们基于书生.浦语的1.8B的模型进行`fine tune`，并利用`Prompt`提示词技术使模型的知识充满隋朝背景，因为`LLM`对于隋朝历史的存在知识匮乏现象，我们在`QA`过程的上游部分利用`Qwen`大模型整合了内外部知识和`ReAct`行动推理技术，极大减缓了低参数模型的幻觉现象。简而言之，我们主要利用了低参数模型的训练，并在低参数的训练结果之上传给高参数模型进行知识重塑，在增强特定领域的知识能力的同时也保留了LLM的能力，并极大地减少了训练成本。其次，如果低参数模型出现过拟合现象时，我们的上游工作部分也能够很好的对回答内容进行知识补充。





#### 2.快速开始

**目录介绍：**

- .pth：模型训练后的文件（在.zip包下）
- .json：训练数据
- xtuner_streamlit_demo：webDemo启动文件
- ReActFromOpen：上游知识填充
- internlm2_chat_1_8b_qlora_alpaca_e3_copy.py：配置文件

**快速开始：**

1. 首先根据[Tutorial/docs/L1/XTuner/readme.md at camp3 · InternLM/Tutorial (github.com)](https://github.com/InternLM/Tutorial/blob/camp3/docs/L1/XTuner/readme.md)安装好相关Xtuner的环境

```shell
# 创建虚拟环境
conda create -n xtuner0121 python=3.10 -y

# 激活虚拟环境（注意：后续的所有操作都需要在这个虚拟环境中进行）
conda activate xtuner0121

# 安装一些必要的库
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖
pip install transformers==4.39.3
pip install streamlit==1.36.0

# 创建一个目录，用来存放源代码
mkdir -p /root/InternLM/code

cd /root/InternLM/code
# 安装
git clone -b v0.1.21  https://github.com/InternLM/XTuner /root/InternLM/code/XTuner
# 查看版本
xtuner version
```

2. 我们已经准备好了`.pth`文件在`.zip`包下,将训练后的.pth进行转换为目前通用的 `HuggingFace` 格式文件，得到`merged`

```shell
# 先获取最后保存的一个pth文件
pth_file=`ls -t *.pth | head -n 1`
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py ${pth_file} ./hf
```

3. 然后进行模型合并即可

```shell
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert merge /root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b ./hf ./merged --max-shard-size 2GB
```

4. 最后启动`xtuner_streamit_demo`文件

```shell
streamlit run xtuner_streamlit_demo.py
```

5. 效果图
   ![image](https://github.com/user-attachments/assets/c0f9a154-aaa2-42c3-b356-df93a58322a6)

![image](https://github.com/user-attachments/assets/55ff610b-64ac-4db5-88eb-e35feb91cfaf)







#### 3.项目结构图

![image](https://github.com/user-attachments/assets/c98c1647-3de7-4c8a-b8a2-956415191f07)
