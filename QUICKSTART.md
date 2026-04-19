# RAG Engine - Quick Start Guide

## 🚀 5分钟快速开始

### 1. 环境准备

```bash
# 复制环境配置
cp .env.example .env

# 编辑 .env 文件，填入您的 OpenAI API 密钥
# OPENAI_API_KEY=sk-your-key-here
```

### 2. 安装依赖

```bash
# 使用 pip 安装
pip install -e .

# 或使用 uv 安装（推荐）
# curl -LsSf https://astral.sh/uv/install.sh | sh
# uv sync
```

### 3. 运行示例

```bash
# 基础示例
python examples/basic_example.py

# 多模态示例
python examples/multimodal_example.py

# 多语言示例
python examples/multilingual_example.py
```

## 📖 核心概念理解

### 什么是 RAG？
Retrieval Augmented Generation（检索增强生成）是一种将信息检索和文本生成结合的技术：
1. **检索 (Retrieval)**: 从知识库中找到相关信息
2. **增强 (Augmentation)**: 用这些信息补充提示
3. **生成 (Generation)**: 基于增强后的提示生成答案

### RAG Engine 的优势

| 特性 | 说明 |
|------|------|
| 🎯 多模态 | 同时处理文本、图像、表格、公式等 |
| 🧠 知识图谱 | 自动构建实体和关系 |
| 🔍 混合检索 | 向量+图谱的混合检索 |
| 🌍 多语言 | 7种语言原生支持 |
| 📦 无依赖 | 不依赖其他 RAG 框架，完全基于 LLamaIndex |

## 💡 常见使用场景

### 场景1: 文档问答系统

```python
# 1. 处理文档
engine.process_folder("./documents/")

# 2. 用户提问
result = engine.query("文档中的主要内容是什么?")

# 3. 获取答案
print(result.answer)
```

### 场景2: 多语言知识库

```python
# 处理英文文档
engine.process_document("english_doc.pdf", language="en")

# 处理中文文档
engine.process_document("chinese_doc.txt", language="zh")

# 双语查询
en_result = engine.query("What is AI?", language="en")
zh_result = engine.query("什么是人工智能?", language="zh")
```

### 场景3: 研究论文分析

```python
# 处理论文
engine.process_document("research_paper.pdf")

# 关键问题
result = engine.query(
    "这篇论文的主要贡献是什么?",
    top_k=5,
    use_graph=True
)

# 获取源文献
for doc in result.retrieved_docs:
    print(f"- {doc.content[:100]}... (score: {doc.score:.2f})")
```

## 🔧 故障排查

### 问题: ImportError

```
ModuleNotFoundError: No module named 'openai'
```

**解决方案**: 安装完整依赖
```bash
pip install -e .
# 或
pip install openai llama-index
```

### 问题: API 密钥错误

```
AuthenticationError: Invalid API key
```

**解决方案**: 检查 .env 文件中的密钥
```bash
# 确认设置正确
cat .env | grep OPENAI_API_KEY

# 测试连接
python -c "from openai import OpenAI; print('OK')"
```

### 问题: 文件解析失败

```
ValueError: No parser found for file
```

**解决方案**: 确保文件格式被支持
- ✅ PDF, DOCX, XLSX, TXT, MD, JPG, PNG
- ❌ 其他格式需要自定义解析器

## 📚 学习路径

1. **初级**: 运行示例，理解基础概念
2. **中级**: 修改配置，处理自己的文档
3. **高级**: 自定义处理器，扩展功能
4. **专家**: 贡献代码，参与开发

## 🎓 深入学习

### 推荐阅读

1. [README.md](README.md) - 项目概览
2. [USAGE_GUIDE.md](USAGE_GUIDE.md) - 详细使用指南
3. [Code Examples](examples/) - 完整代码示例

### 相关资源

- [LLamaIndex 文档](https://docs.llamaindex.ai)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [RAG 论文和研究](https://arxiv.org/)

## 💬 获取帮助

### 联系方式

- 📧 提交 Issue: 在 GitHub 上报告问题
- 💬 讨论: 参与社区讨论
- 📖 文档: 查看详细文档

## 🎉 下一步

现在您已经了解了基础知识，可以：

1. ✅ 处理您的第一个文档
2. ✅ 尝试多种查询方式
3. ✅ 探索多语言功能
4. ✅ 自定义配置参数
5. ✅ 贡献您的想法和代码

祝您使用愉快！ 🚀
