
# MCM多模态中医问诊大模型

<div style="text-align: center;">
  <img src="https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2024-06-22-logo_MCM-removebg-preview.png" style="zoom:30%;" />
</div>

[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)

## 📝目录

- [📖 简介](#intro)
- [🚀News](#News)
- [😈模型架构](#模型架构)
  - [📊数据与微调](#数据与微调)
  - [🧀知识图谱构建](#知识图谱构建)
  - [🩺可解释问诊](#可解释问诊)
  - [🏓智能评测](#智能评测)

- [🛠️ 使用方法](#使用方法)
- [💕 项目成员](#项目成员)
- [🖊️ Citation](#Citation)
- [🉑开源许可证](#开源许可证)

## 📖 简介 <a id="intro"></a>

MCM(**M**ultimodal **C**hinese **M**edical LLM)是由上海计算机软件技术开发中心研发的多模态中医药问诊大模型，该模型可以通过与用户的对话进行问诊，问诊过程支持医学影像处理，问诊过程由**知识图谱驱动，具备可解释性**，同时支持常用中医知识问答。

## 🚀News <a id="News"></a>

[2024.06.22] MCM 第一版上线！

## 😈模型架构 <a id="模型架构"></a>

![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2024-06-22-080226.png)

### 📊数据与微调 <a id="数据与微调"></a>

我们基于[XTuner](https://github.com/InternLM/xtuner)进行了知识注入。搜集了医学开源数据集、中医典籍等语料，并通过这些语料对[InternLM2-20B](https://huggingface.co/internlm/internlm2-20b)模型进行了增量预训练(Continuous Pre-train, CPT)；然后，通过基于中文问题生成的开源预训练模型[MT5](https://huggingface.co/algolet/mt5-base-chinese-qg)生成生成QA对，并结合现有开源中医多轮对话QA对，进行了有监督微调（Supervised Fine-tune，SFT），最终实现了针对医疗知识的知识注入和问诊模式微调。

### 🧀知识图谱构建 <a id="知识图谱构建"></a>

我们利用大模型将所搜集的中医典籍中的文言文翻译为现代汉语；然后利用[OneKE](https://huggingface.co/openkg/OneKE)知识抽取大模型对翻译后的中医典籍进行三元组抽取，抽取了包括疾病、方剂、临床表现、病因、药材、功效、性味、入药部位、用法用量、注意事项等10种实体与12种关系，并利用neo4j图数据库将这些三元组构建成知识图谱。

![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2024-06-22-074931.png)

### 🩺可解释问诊 <a id="可解释问诊"></a>

由于医学问诊场景的特殊性，为实现切实可落地，我们构建了一个可解释问诊Agent。该Agent后端由多源数据检索RAGAgent和医学影像处理大模型[MedDr](https://github.com/sunanhe/MedDr)联合驱动，在问诊过程中可将思考过程输出，并构建了高效的上下文维护机制，具有对无关对话的鲁棒性。

### 🏓智能评测 <a id="智能评测"></a>

本项目的智能评测采用了团队开发的大语言模型全栈式自动测评工具箱[GreatLibrarian](https://github.com/JerryMazeyu/GreatLibrarian.git)。首先获取医学领域的benchmark数据集，将其转换为开放式问题，然后使用测评工具箱[GreatLibrarian](https://github.com/JerryMazeyu/GreatLibrarian.git)对微调后的模型进行自动化测评，生成测评报告，在整个开发过程中可以对性能进行随时监控。

## 🛠️ 使用方法 <a id="使用方法"></a>

* 下载项目并添加路径

```shell
git clone <GithubRepo>
export PYTHONPATH=\$PYTHONPATH:<Local Path>
```

* 安装Conda环境

```shell
conda create -n MCM python=3.10
conda activate MCM
python -m pip install -r requirements.txt
```

* 运行Neo4j知识图谱

*注：Neo4j 5.x需要安装jdk17，以下第一行命令为在ubuntu下安装jdk17，若已安装可跳过，windows或mac os可前往官网下载安装包（https://www.oracle.com/java/technologies/downloads/archive/ ）*

```shell
sudo apt-get install openjdk-17-jdk  # 若已安装则可跳过
cd Dist/libs/neo4j-community-5.20.0/bin
neo4j start
```

* 运行Demo

```shell
python demo.py
```

## 💕 项目成员 <a id="项目成员"></a>

- [马泽宇](https://github.com/JerryMazeyu)（上海软件中心 人工智能研究与测评部 整体项目规划、代码架构、数据搜集）
- [王婉莹](https://github.com/tiezhuguangtailang)（上海软件中心 人工智能研究与测评部 多模态Agent、RAG Agent构建、数据搜集）
- [丁敏捷](https://github.com/ggxxding)（上海软件中心 人工智能研究与测评部 数据预处理、知识图谱构建）
- [梁晨诞](https://github.com/LiangRichard13) （上海计算所&第二工业大学 计算机与信息工程学院 数据预处理、知识注入及模型微调）
- [曹致远](https://github.com/EnumaElish123)（上海计算所&第二工业大学 计算机与信息工程学院 数据预处理及智能评测）
- [李翔](https://github.com/LX945794759)（上海计算所&第二工业大学 计算机与信息工程学院 版本管理及运维）

## 🖊️ Citation <a id="Citation"></a>

```bibtex
@misc{2024MCM,
    title={MCM: Multimodal Chinese Medical Large Model},
    author={MCM Contributors},
    howpublished = {\url{https://github.com/JerryMazeyu/MCM}},
    year={2024}
}
```

## 🉑开源许可证 <a id="开源许可证"></a>

该项目采用 [Apache License 2.0 开源许可证](https://github.com/AXYZdong/AMchat/blob/main/LICENSE) 同时，请遵守所使用的模型与数据集的许可证。
