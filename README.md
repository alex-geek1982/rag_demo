# RAG Engine - All-in-One Multimodal RAG Framework

> 基于 LLamaIndex 的企业级RAG引擎，完全复制 RAG-Anything 的能力，原生多语言支持

## ✨ 核心特性

### 🎯 多模态处理
- **文本分析**: 自动提取和处理文本内容
- **图像处理**: 支持图像识别和VLM分析
- **表格处理**: 结构化数据解析和索引
- **数学公式**: LaTeX公式识别与关联
- **代码块**: 程序代码分析与理解

### 📄 文档支持
支持以下格式的自动解析：
- PDF 文件
- Word 文档 (.docx, .doc)
- Excel 表格 (.xlsx, .xls)
- 图像格式 (.jpg, .png, .gif 等)
- 文本文件 (.txt, .md)

### 🧠 智能知识图谱
- **实体提取**: 自动从文档中提取关键实体
- **关系构建**: 建立多模态内容之间的关系
- **图谱遍历**: 支持复杂的关系查询和推理
- **跨模态链接**: 连接不同类型的内容

### 🔍 混合检索
- **向量检索**: 基于语义的相似度搜索
- **知识图谱遍历**: 基于实体关系的检索
- **模态过滤**: 按内容类型过滤结果
- **相关性排序**: 智能排序和评分

### 🌍 真正的多语言支持
不仅仅是界面翻译，而是**完整的多语言检索和生成**：

- **自动语言检测**: 自动检测文档和查询的语言（7+种语言）
- **多语言嵌入**: 统一的向量空间支持跨语言检索
- **跨语言检索**: 用一种语言查询，检索其他语言的相关文档
- **语言亲和度评分**: 同语言匹配获得15%的相关性提升
- **元数据追踪**: 保留整个处理流程中的语言信息
- **支持语言**: 英文、中文、日文、韩文、西班牙文、法文、德文

**示例**：用英文查询，自动获取英文、中文、日文等多语言的相关结果

详见：[多语言RAG功能说明](MULTILINGUAL_RAG.md)

## 🚀 快速开始

### 安装
```bash
# 基础安装
pip install -e .

# 开发环境
pip install -e ".[dev]"
```

### 环境配置
```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 配置 OpenAI API 密钥
OPENAI_API_KEY=your_api_key_here
```

### 基础使用
```python
from rag_engine.core import create_engine
from rag_engine.config import RAGEngineConfig

# 创建引擎
config = RAGEngineConfig.from_env()
engine = create_engine(config)

# 处理文档
doc = engine.process_document("path/to/document.pdf")

# 查询
result = engine.query("What is the main topic?")
print(result.answer)
```

## 📚 详细使用指南

### 1. 文档处理

#### 单个文档
```python
doc = engine.process_document(
    "report.pdf",
    doc_id="report_001",
    doc_title="Annual Report",
    language="en"
)
```

#### 批量处理
```python
documents = engine.process_folder(
    "documents/",
    language="en",
    recursive=True
)
```

### 2. 多模态查询

#### 基础查询
```python
result = engine.query(
    "What are the key findings?",
    top_k=5
)
```

#### 异步查询
```python
result = await engine.aquery(
    "What are the key findings?",
    top_k=5
)
```

### 3. 多语言处理

```python
from rag_engine.i18n import set_language

# 切换语言
set_language("zh")  # 中文
set_language("en")  # 英文
set_language("ja")  # 日文

# 按特定语言处理文档
doc = engine.process_document(
    "chinese_doc.txt",
    language="zh"
)
```

### 4. 知识图谱操作

```python
# 获取知识图谱统计
stats = engine.kg.get_stats()

# 搜索实体
entities = engine.kg.search_entities_by_name("AI")

# 获取相关实体
related = engine.kg.get_related_entities("entity_id")

# 查找实体路径
paths = engine.kg.get_entity_paths("start_id", "end_id")

# 保存知识图谱
engine.kg.save("kg.json")
```

### 5. 检索统计

```python
stats = engine.get_statistics()
print(stats)
# {
#     'processed_documents': 10,
#     'total_content_blocks': 150,
#     'knowledge_graph': {...},
#     'retriever': {...}
# }
```

## 🏗️ 架构设计

### 系统架构流程

```
文档输入 → 文档解析 → 内容处理 → 知识图谱 → 检索引擎 → 答案生成
  ↓        ↓          ↓          ↓          ↓         ↓
多格式    多解析器    多处理器    图构建     混合检索   LLM集成
支持      器         器          和索引     器
```

### 核心模块

1. **parsers/** - 文档解析
   - `PDFParser`: PDF 解析
   - `DocxParser`: Word 文档解析
   - `ExcelParser`: Excel 表格解析
   - `ImageParser`: 图像处理
   - `TextParser`: 纯文本解析

2. **processors/** - 内容处理
   - `TextProcessor`: 文本处理
   - `ImageProcessor`: 图像分析
   - `TableProcessor`: 表格解析
   - `EquationProcessor`: 公式处理
   - `CodeProcessor`: 代码分析

3. **core/** - 核心引擎
   - `RAGEngine`: 主引擎
   - `KnowledgeGraph`: 知识图谱
   - `EntityExtractor`: 实体提取
   - `RelationshipBuilder`: 关系构建

4. **retrieval/** - 检索系统
   - `HybridRetriever`: 混合检索
   - `OpenAIEmbedding`: 文本嵌入
   - 向量相似度搜索

5. **i18n/** - 国际化
   - 多语言支持
   - 自定义翻译

## 📋 API 参考

### RAGEngine

#### 核心方法

```python
# 处理单个文档
process_document(file_path, doc_id, doc_title, language) -> Document

# 批量处理文件夹
process_folder(folder_path, language, recursive) -> List[Document]

# 同步查询
query(query, top_k, use_graph) -> QueryResult

# 异步查询
async aquery(query, top_k, use_graph) -> QueryResult

# 获取统计信息
get_statistics() -> Dict

# 保存引擎状态
save_state(output_dir) -> str
```

### QueryResult

```python
@dataclass
class QueryResult:
    query: str                          # 原始查询
    answer: str                         # 生成的答案
    retrieved_docs: List[RetrievalResult]  # 检索结果
    sources: List[str]                  # 信息来源
    confidence: float                   # 置信度
    reasoning: Optional[str]            # 推理过程
    multimodal_analysis: Optional[Dict] # 多模态分析结果
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_parsers.py

# 查看覆盖率
pytest --cov=rag_engine tests/
```

## 📝 示例

### 1. 基础示例
```bash
python examples/basic_example.py
```

### 2. 多模态处理
```bash
python examples/multimodal_example.py
```

### 3. 多语言支持
```bash
python examples/multilingual_example.py
```

## 🔧 配置项

### 环境变量

```bash
# OpenAI API
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=https://api.openai.com/v1

# Gemini Vision API (可选)
VISION_PROVIDER=gemini          # 或 "openai"（默认）
GEMINI_API_KEY=AIza...          # Gemini API 密钥

# 模型配置
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7

# 存储设置
WORKING_DIR=./rag_storage
OUTPUT_DIR=./output

# 处理配置
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
MAX_WORKERS=4
ENABLE_MULTIMODAL=true

# 语言设置
LANGUAGE=en
```

### Vision 模型支持

本项目支持多种 Vision API 提供商用于 PDF 图像分析和多模态处理：

#### OpenAI Vision (默认)
```bash
VISION_PROVIDER=openai
OPENAI_API_KEY=sk-...
# 使用 GPT-4V、GPT-4o 等模型
```

#### Google Gemini Vision
```bash
VISION_PROVIDER=gemini
GEMINI_API_KEY=AIza...
# 使用 Gemini 2.5 Flash、Gemini Pro Vision 等模型
```

**快速开始 Gemini**:
```bash
pip install google-generativeai
python test_gemini_config.py
python examples/gemini_pdf_rag_example.py
```

详见：[Gemini Vision Setup Guide](GEMINI_VISION_SETUP.md) 和 [实现总结](GEMINI_IMPLEMENTATION_SUMMARY.md)

## 🎨 定制化

### 自定义处理器

```python
from rag_engine.processors import BaseModalProcessor

class CustomProcessor(BaseModalProcessor):
    def supports(self, content_type):
        return content_type == ContentType.CUSTOM
    
    def process(self, block):
        # 自定义处理逻辑
        description = "..."
        entity = Entity(...)
        return description, entity
```

### 自定义翻译

```python
from rag_engine.i18n import get_i18n

i18n = get_i18n()
i18n.add_language("custom", {
    "key1": "value1",
    "key2": "value2"
})
```

## 📊 功能对比

| 功能 | RAG-Anything | RAG Engine |
|------|-------------|-----------|
| 多模态处理 | ✅ | ✅ |
| 知识图谱 | ✅ | ✅ |
| 混合检索 | ✅ | ✅ |
| 多语言支持 | ⚠️ | ✅ |
| LLamaIndex 基础 | ❌ | ✅ |
| 无前端 | ✅ | ✅ |
| 易于扩展 | ⚠️ | ✅ |

## 🤝 扩展指南

### 添加新的文档格式

1. 在 `parsers/` 中创建新的解析器类
2. 继承 `BaseParser`
3. 实现 `parse()` 和 `supports()` 方法
4. 在 `ParserFactory` 中注册

### 添加新的处理器

1. 在 `processors/` 中创建新的处理器类
2. 继承 `BaseModalProcessor`
3. 实现 `process()` 和 `supports()` 方法
4. 在 `ProcessorFactory` 中注册

## 📄 许可证

MIT

## 🙋 支持

如有问题或建议，欢迎提交 Issue 或 PR。
