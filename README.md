# Intelligent-Customer-Service

一个基于多Agent协作的智能客服系统，集成知识库导入管理、智能对话问答等核心能力，为企业提供高效、准确、全天候的客户服务解决方案。

## 项目简介

本项目采用多Agent架构设计，将复杂的客服流程拆解为多个专业化Agent协同完成。系统支持从PDF等文档自动导入并构建知识库，通过向量检索与大语言模型结合的方式实现精准问答，并在需要时将复杂问题无缝转接给人工客服。

## 核心功能

### 知识库导入与管理

- **文档解析**：支持 PDF 等格式的文档自动解析，将非结构化内容转换为可检索的知识条目
- **Markdown 转换**：文档内容自动转换为结构化 Markdown 格式
- **图片识别**：支持文档中图片内容的提取与识别
- **文档切分**：智能文档分割，按语义单元拆分为独立知识片段
- **条目识别**：自动识别并提取知识条目名称与分类信息
- **向量化存储**：基于 BGE 模型生成文本嵌入，导入 Milvus 向量数据库实现高效检索

### 智能对话 / 问答

- **意图识别**：准确理解用户问题意图，匹配最相关的知识内容
- **多轮对话**：支持上下文关联的连续对话，提供流畅的交互体验
- **知识检索增强生成（RAG）**：结合向量检索与大语言模型，给出准确且自然的回答

### 多Agent协作

- **任务分发**：根据用户请求类型自动路由到对应的专业 Agent
- **流程编排**：基于 LangGraph 的工作流编排，实现灵活的 Agent 协作流程
- **条件路由**：支持基于状态的条件分支，动态决定处理路径

## 技术架构

```
用户请求
  │
  ▼
入口节点 (node_entry)
  │
  ├── PDF 解析 → Markdown 转换 → 图片识别 → 文档切分 → 条目识别 → BGE 向量化 → Milvus 导入
  │
  └── 智能问答 → 知识检索 → LLM 生成回答
```

## 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangGraph |
| 向量数据库 | Milvus |
| 嵌入模型 | BGE Embedding |
| 后端框架 | FastAPI |
| 编程语言 | Python |

## 项目结构

```
Intelligent-Customer-Service/
├── dataset-agent/
│   ├── app/
│   │   └── import_process/
│   │       └── agent/
│   │           └── main_graph.py      # 主流程图定义
│   ├── doc/                           # 文档资源
│   └── output/                        # 输出目录
└── README.md
```

## 快速开始

### 环境要求

- Python 3.10+
- Milvus 向量数据库
- 相关 API Key（大语言模型服务）

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/yyifeng145-afk/Intelligent-Customer-Service.git
cd Intelligent-Customer-Service

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

```bash
python -m dataset-agent.app.main
```

## 工作流程

1. **文档导入**：上传 PDF 等格式的设备手册或产品文档
2. **内容解析**：系统自动完成 PDF 解析、Markdown 转换、图片识别
3. **知识构建**：文档切分后进行条目识别，通过 BGE 模型向量化并存入 Milvus
4. **智能问答**：用户提问时，系统从知识库中检索相关内容，结合 LLM 生成准确回答

## 许可证

MIT License
