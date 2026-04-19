# 新架构设计文档

## 架构概览

### 分层架构

```
┌─────────────────────────────────────────────────┐
│           Application & Examples                │
│  (examples/hybrid_pdf_rag_chroma_kuzu.py)      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│           Pipeline Layer (新)                   │
│ ┌──────────────────────────────────────────┐  │
│ │ DocumentProcessor   → content blocks     │  │
│ │ KnowledgeBaseBuilder → embeddings       │  │
│ │ KnowledgeGraphBuilder → entities/rels   │  │
│ │ RetrievalPipeline → results/answer      │  │
│ └──────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│           Storage Layer (新)                    │
│ ┌──────────────────────────────────────────┐  │
│ │ ChromaKnowledgeBase    (独立)            │  │
│ │ KuzuGraphStore         (独立)            │  │
│ └──────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│        External Services (已有)                 │
│ ┌──────────────────────────────────────────┐  │
│ │ Chroma DB | Kuzu DB | LLM | Embedding   │  │
│ └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## 模块详情

### Pipeline Layer

#### DocumentProcessor
```python
DocumentProcessor
  ├── process_document(file_path) → Document
  ├── process_folder(folder_path) → List[Document]
  └── _process_content_blocks(document)
```

依赖：
- ParserFactory (已有)
- ProcessorFactory (已有)

职责：
- 解析各种格式的文档
- 提取内容块
- 应用处理器进行块增强

特点：
- 单纯的文件输入，没有副作用
- 返回结构化的 Document 对象
- 支持多种文件格式

#### KnowledgeBaseBuilder
```python
KnowledgeBaseBuilder
  ├── build_from_document(document)
  ├── build_from_blocks(blocks)
  ├── rebuild_chroma(chroma_kb)
  ├── get_embeddings() → Dict[str, List[float]]
  └── get_content_blocks() → Dict[str, ContentBlock]
```

依赖：
- OpenAIEmbedding (已有)

职责：
- 生成向量嵌入
- 管理内容块和向量的对应关系
- 协调 Chroma DB 重建

特点：
- 所有计算都是可观测的
- 支持分步操作（先构建，后重建）
- 嵌入向量完全可访问

#### KnowledgeGraphBuilder
```python
KnowledgeGraphBuilder
  ├── build_from_document(document)
  ├── build_from_blocks(blocks)
  ├── rebuild_kuzu(kuzu_store)
  ├── rebuild_kuzu_from_chroma_chunks(kuzu_store, chunks)  # 🆕
  ├── get_entities() → Dict[str, Entity]
  ├── get_relationships() → Dict[str, Relationship]
  └── get_graph_stats() → Dict
```

依赖：
- EntityExtractor (已有)
- RelationshipBuilder (已有)
- KnowledgeGraph (已有)

职责：
- 提取实体和关系
- 管理知识图谱
- 协调 Kuzu DB 重建

特点：
- ✨ 支持两种重建方式（从 document 或从 Chroma）
- 实体和关系完全可访问
- 支持图统计信息查询

#### RetrievalPipeline
```python
RetrievalPipeline
  ├── retrieve_hybrid(query, chroma_kb, kuzu_store, ...) → List[RetrievalResult]
  └── run_query(query, chroma_kb, kuzu_store, ...) → Dict
```

包含：
- LocalReranker
- LocalAnswerGenerator

职责：
- 协调向量和图搜索
- 合并和去重结果
- Reranking
- 答案生成

特点：
- 完全解耦的检索组件
- 支持灵活的排序和生成配置

### Storage Layer

#### ChromaKnowledgeBase
```python
ChromaKnowledgeBase
  ├── __init__(db_path)
  ├── get_or_create_collection(name)
  ├── rebuild(content_blocks, embeddings, collection_name)
  ├── search(query_vector, top_k, collection_name) → List[RetrievalResult]
  ├── count() → int
  └── get_all() → Dict (用于 KG 重建)
```

特点：
- ✅ 完全独立，不依赖任何其他组件
- ✅ 可以单独初始化和使用
- ✅ 支持多个 collection
- ✅ 持久化存储
- ✅ 提供数据导出接口

设计原理：
- 关注点分离：只处理向量存储
- 接口简洁：只暴露必要的方法
- 状态管理：内部维护集合的轻量级状态

#### KuzuGraphStore
```python
KuzuGraphStore
  ├── __init__(db_path)
  ├── rebuild_from_entities_and_relationships(entities, relationships, blocks)
  ├── rebuild_from_chroma_chunks(chunk_ids, documents, metadatas)  # 🆕
  ├── search(query, entities, content_blocks, top_k) → List[RetrievalResult]
  ├── get_or_create_collection(name)
  └── _escape(value) → str
```

特点：
- ✅ 完全独立，不依赖任何其他组件
- ✅ 支持两种重建方式
- ✨ 新特性：可以从 Chroma 数据直接重建（无需文档处理！）
- ✅ 基于关键词和邻接的混合搜索
- ✅ 持久化存储

设计原理：
- 灵活的重建机制
- 支持多种数据源
- 查询优化（支持邻接遍历）

## 数据流

### 工作流 A: 完整处理（UPDATE_KB）
```
PDF File
   │
   ▼
DocumentProcessor
   │ → Document (with content blocks)
   │
   ├─────────────────────┬───────────────────┐
   │                     │                   │
   ▼                     ▼                   ▼
KBBuilder            KGBuilder          (data collection)
   │                    │
   ├─ embeddings        ├─ entities
   └─ content blocks    └─ relationships
   │                    │
   ▼                    ▼
ChromaKnowledgeBase  KuzuGraphStore
   │                    │
   ▼                    ▼
Chroma DB            Kuzu DB
```

### 工作流 B: 独立 KG 重建（UPDATE_KG）✨
```
Chroma DB
   │
   ▼
ChromaKnowledgeBase.get_all()
   │ → {chunk_ids, documents, metadatas}
   │
   ▼
KnowledgeGraphBuilder.rebuild_kuzu_from_chroma_chunks()
   │
   ▼
KuzuGraphStore
   │
   ▼
Kuzu DB

✨ 优势：
- 无需重新处理 PDF
- 无需重新生成 embeddings
- 只需要 Chroma 中已有的数据
- 支持快速迭代 KG 构建逻辑
```

### 工作流 C: 查询执行（EXECUTE_QUERY）
```
Query
   │
   ▼
RetrievalPipeline.run_query()
   │
   ├─────────────────┬────────────────┐
   │                 │                │
   ▼                 ▼                ▼
ChromaKnowledgeBase KuzuGraphStore (metadata)
   │                 │                │
   └─────────────┬───┘                │
                 │                    │
                 ▼                    ▼
         Merged Results        LocalReranker
                 │
                 ▼
         LocalAnswerGenerator
                 │
                 ▼
            Answer
```

## 关键设计决策

### 1. 完全的存储层解耦
**决策**: ChromaKnowledgeBase 和 KuzuGraphStore 完全独立
**理由**:
- 支持单独初始化和使用
- 便于替换实现（如 Pinecone、Neo4j）
- 更容易测试

**示例**:
```python
# 直接使用，无需 engine
chroma = ChromaKnowledgeBase(path)
results = chroma.search(vector, top_k=5)
```

### 2. 两种 KG 重建方式
**决策**: `rebuild_kuzu()` 和 `rebuild_kuzu_from_chroma_chunks()`
**理由**:
- 支持从不同数据源重建
- 不同的使用场景
- 无需重复处理标记

**示例**:
```python
# 方式 1：从实体/关系重建
kg_builder.rebuild_kuzu(kuzu_store)

# 方式 2：从 Chroma 数据重建（新特性！）
kg_builder.rebuild_kuzu_from_chroma_chunks(kuzu_store, chunks)
```

### 3. Pipeline 组件的独立性
**决策**: DocumentProcessor, KBBuilder, KGBuilder 可单独使用
**理由**:
- 支持灵活的工作流
- 每个组件可以独立测试
- 支持增量更新

**示例**:
```python
# 只处理文档，不构建 KB/KG
processor = DocumentProcessor(config)
doc = processor.process_document("file.pdf")

# 只构建 KB，不构建 KG
kb_builder = KnowledgeBaseBuilder(config)
kb_builder.build_from_document(doc)

# 独立存储
chroma.rebuild(kb_builder.get_content_blocks(), kb_builder.get_embeddings())
```

## 扩展点

### 添加新的存储后端
1. 继承 `ChromaKnowledgeBase` 的接口
2. 实现 `rebuild()` 和 `search()`
3. 更新 `KnowledgeBaseBuilder.rebuild_*()` 方法

### 添加新的检索策略
1. 在 `RetrievalPipeline` 中添加新方法
2. 支持不同的合并和排序策略
3. 可选的重新排序器

### 自定义 KG 构建
1. 继承 `KnowledgeGraphBuilder`
2. 重写 `build_from_blocks()` 方法
3. 自定义实体和关系提取

## 性能特性

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| 文档处理 | O(n) | n = 文本大小 |
| 嵌入生成 | O(n * d) | d = embedding 维度 |
| 向量搜索 | O(log n) | HNSW 索引 |
| 图搜索 | O(e) | e = 输出结果数 |
| 重新排序 | O(n log n) | n = 候选结果数 |

## 向后兼容性

所有现有的 `RAGEngine` API 仍然有效：
```python
# OLD API - 继续支持
engine = RAGEngine(config)
engine.process_document("file.pdf")
engine.build_knowledge_graph_for_document(document)
engine.index_content_blocks(blocks)
engine.query("question")

# 内部实现已委托给新模块，但 API 保持不变
```

## 迁移路径

1. **立即（无需修改）**：现有代码继续使用 RAGEngine API
2. **短期（推荐）**：尝试模块化 API，享受灵活性
3. **长期（最佳实践）**：完全使用模块化架构

