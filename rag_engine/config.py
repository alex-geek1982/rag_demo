"""
RAG Engine - Configuration management
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    dimension: int = int(os.getenv("EMBEDDING_DIM", "3072"))
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    batch_num: int = int(os.getenv("EMBEDDING_BATCH_NUM", "32"))  # Batch size for embedding
    max_async: int = int(os.getenv("EMBEDDING_MAX_ASYNC", "10"))  # Concurrent embedding requests
    cache_config: Optional[Dict[str, Any]] = None  # Cache configuration
    # Azure OpenAI support
    use_azure: bool = os.getenv("EMBEDDING_USE_AZURE", "false").lower() == "true"
    azure_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version: Optional[str] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    azure_deployment: Optional[str] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


@dataclass
class LLMConfig:
    """LLM configuration"""
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_max_token_size: int = int(os.getenv("LLM_MODEL_MAX_TOKEN_SIZE", "8192"))
    model_max_async: int = int(os.getenv("LLM_MODEL_MAX_ASYNC", "10"))
    model_kwargs: Dict[str, Any] = field(default_factory=dict)  # Additional model parameters
    enable_cache: bool = os.getenv("LLM_ENABLE_CACHE", "true").lower() == "true"  # Enable LLM response caching
    # Azure OpenAI support
    use_azure: bool = os.getenv("LLM_USE_AZURE", "false").lower() == "true"
    azure_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version: Optional[str] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    azure_deployment: Optional[str] = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")


@dataclass
class VisionConfig:
    """Vision model configuration for multimodal processing"""
    model: str = os.getenv("VISION_MODEL", "gpt-4o")
    temperature: float = float(os.getenv("VISION_TEMPERATURE", "0.7"))
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True
    provider: str = os.getenv("VISION_PROVIDER", "openai")  # "openai", "azure", or "gemini"
    # Azure OpenAI support
    azure_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version: Optional[str] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    azure_deployment: Optional[str] = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT")


@dataclass
class LanguageConfig:
    """Multi-language support configuration"""
    default_language: str = os.getenv("LANGUAGE", "en")
    supported_languages: List[str] = field(default_factory=lambda: ["en", "zh", "ja", "ko", "es", "fr", "de"])
    enable_translation: bool = True


@dataclass
class PDFProcessingConfig:
    """Advanced PDF processing configuration"""
    use_advanced_layout: bool = os.getenv("PDF_USE_ADVANCED_LAYOUT", "true").lower() == "true"
    extract_images: bool = os.getenv("PDF_EXTRACT_IMAGES", "true").lower() == "true"
    extract_tables: bool = os.getenv("PDF_EXTRACT_TABLES", "true").lower() == "true"
    extract_text: bool = os.getenv("PDF_EXTRACT_TEXT", "true").lower() == "true"
    use_vision_api: bool = os.getenv("PDF_USE_VISION_API", "true").lower() == "true"
    vision_api_key: Optional[str] = os.getenv("PDF_VISION_API_KEY")  # Defaults to OPENAI_API_KEY if not set
    vision_base_url: Optional[str] = os.getenv("PDF_VISION_BASE_URL")  # Vision API endpoint
    vision_model: Optional[str] = os.getenv("PDF_VISION_MODEL")  # Vision model name
    vision_provider: str = os.getenv("PDF_VISION_PROVIDER", "openai")  # "openai", "azure", or "gemini"
    # Azure OpenAI fields for vision (used when vision_provider == "azure")
    vision_azure_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    vision_azure_api_version: Optional[str] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    vision_azure_deployment: Optional[str] = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT")
    context_window_pixels: int = int(os.getenv("PDF_CONTEXT_WINDOW_PIXELS", "200"))
    min_image_area: int = int(os.getenv("PDF_MIN_IMAGE_AREA", "1000"))
    max_surrounding_text_chars: int = int(os.getenv("PDF_MAX_SURROUNDING_TEXT_CHARS", "2000"))
    filter_header_footer: bool = os.getenv("PDF_FILTER_HEADER_FOOTER", "true").lower() == "true"
    header_margin_ratio: float = float(os.getenv("PDF_HEADER_MARGIN_RATIO", "0.08"))
    footer_margin_ratio: float = float(os.getenv("PDF_FOOTER_MARGIN_RATIO", "0.08"))
    header_footer_min_repeat_pages: int = int(os.getenv("PDF_HEADER_FOOTER_MIN_REPEAT_PAGES", "2"))
    save_images: bool = os.getenv("PDF_SAVE_IMAGES", "true").lower() == "true"
    
    @classmethod
    def from_vision_config(cls, vision_config: "VisionConfig", **overrides: Any) -> "PDFProcessingConfig":
        """Create PDFProcessingConfig with vision params from VisionConfig"""
        params = {
            "vision_api_key": vision_config.api_key,
            "vision_base_url": vision_config.base_url,
            "vision_model": vision_config.model,
            "vision_provider": vision_config.provider,
            "vision_azure_endpoint": vision_config.azure_endpoint,
            "vision_azure_api_version": vision_config.azure_api_version,
            "vision_azure_deployment": vision_config.azure_deployment,
        }
        params.update(overrides)
        return cls(**params)


@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    # Chunking strategy: 'title' (default) or 'token'
    # 'title': TitleChunker - preserves document hierarchy and structure
    # 'token': TokenChunker - splits by token count with uniform sizes
    chunker_type: str = os.getenv("CHUNKER_TYPE", "title")
    
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    chunk_token_size: int = int(os.getenv("CHUNK_TOKEN_SIZE", "1024"))  # Token-based chunking
    chunk_overlap_token_size: int = int(os.getenv("CHUNK_OVERLAP_TOKEN_SIZE", "20"))
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    max_parallel_insert: int = int(os.getenv("MAX_PARALLEL_INSERT", "4"))  # Parallel inserts to storage
    enable_multimodal: bool = os.getenv("ENABLE_MULTIMODAL", "true").lower() == "true"
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    # Token management
    max_entity_tokens: int = int(os.getenv("MAX_ENTITY_TOKENS", "3000"))  # Max tokens for entities
    max_relation_tokens: int = int(os.getenv("MAX_RELATION_TOKENS", "1000"))  # Max tokens for relations
    max_total_tokens: int = int(os.getenv("MAX_TOTAL_TOKENS", "4000"))  # Max total tokens


@dataclass
class BM25Config:
    """BM25 full-text search configuration"""
    enable_bm25: bool = os.getenv("ENABLE_BM25", "true").lower() == "true"
    k1: float = float(os.getenv("BM25_K1", "1.5"))  # Saturation parameter
    b: float = float(os.getenv("BM25_B", "0.75"))  # Document length normalization
    min_token_length: int = int(os.getenv("BM25_MIN_TOKEN_LENGTH", "2"))  # Min token length for indexing
    language: str = os.getenv("BM25_LANGUAGE", "english")  # For stemming/tokenization


@dataclass
class HybridRetrievalConfig:
    """Hybrid retrieval fusion configuration"""
    # Fusion strategy: 'weighted_avg', 'rrf', 'max', 'min'
    fusion_strategy: str = os.getenv("FUSION_STRATEGY", "weighted_avg")
    
    # Weights for different retrieval methods (sum should be 1.0)
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.5"))
    graph_weight: float = float(os.getenv("GRAPH_WEIGHT", "0.0"))  # Optional graph weight
    
    # Score normalization method: 'minmax', 'sigmoid', 'rank'
    normalization_method: str = os.getenv("NORMALIZATION_METHOD", "minmax")
    
    # RRF (Reciprocal Rank Fusion) parameter
    rrf_k: float = float(os.getenv("RRF_K", "60.0"))
    
    # For rank-based normalization
    rank_offset: float = float(os.getenv("RANK_OFFSET", "1.0"))
    
    # Deduplication: merge results from different sources by doc_id
    enable_dedup: bool = os.getenv("ENABLE_DEDUP", "true").lower() == "true"
    dedup_threshold: float = float(os.getenv("DEDUP_THRESHOLD", "0.95"))  # Similarity threshold
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total_weight = self.vector_weight + self.bm25_weight + self.graph_weight
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            logger_msg = f"Warning: Weights sum to {total_weight}, not 1.0. Normalizing automatically."
            # Auto-normalize
            if total_weight > 0:
                self.vector_weight /= total_weight
                self.bm25_weight /= total_weight
                self.graph_weight /= total_weight


@dataclass
class RerankerConfig:
    """Reranker configuration for result re-ranking"""
    enable_rerank: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
    rerank_model: str = os.getenv("RERANK_MODEL", "simple")  # 'simple', 'cross-encoder', 'llm', 'hybrid'
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "10"))  # Original top-k before reranking
    rerank_final_k: int = int(os.getenv("RERANK_FINAL_K", "5"))  # Final top-k after reranking
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
    device: str = os.getenv("RERANK_DEVICE", "cpu")  # 'cpu' or 'cuda'
    batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "10"))


@dataclass
class StorageConfig:
    """Storage configuration for KV, graph, and vector stores"""
    kv_storage_type: str = os.getenv("KV_STORAGE_TYPE", "json")  # 'json', 'redis', 'sql'
    graph_storage_type: str = os.getenv("GRAPH_STORAGE_TYPE", "json")  # 'json', 'neo4j', 'networkx'
    vector_storage_type: str = os.getenv("VECTOR_STORAGE_TYPE", "json")  # 'json', 'milvus', 'weaviate'
    # Storage parameters
    kv_storage_params: Dict[str, Any] = field(default_factory=dict)
    graph_storage_params: Dict[str, Any] = field(default_factory=dict)
    vector_storage_params: Dict[str, Any] = field(default_factory=dict)
    # Similarity threshold for retrieval
    cosine_threshold: float = float(os.getenv("COSINE_THRESHOLD", "0.3"))
    # Related chunk parameters
    related_chunk_number: int = int(os.getenv("RELATED_CHUNK_NUMBER", "5"))
    # Graph nodes limit
    max_graph_nodes: int = int(os.getenv("MAX_GRAPH_NODES", "10000"))


@dataclass
class RAGEngineConfig:
    """Main RAG Engine configuration"""
    working_dir: str = os.getenv("WORKING_DIR", "./rag_storage")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    pdf_processing: PDFProcessingConfig = field(default_factory=PDFProcessingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    hybrid_retrieval: HybridRetrievalConfig = field(default_factory=HybridRetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Additional settings
    enable_cache: bool = True
    enable_persistence: bool = True
    debug: bool = False
    
    # Advanced features from LightRAG
    enable_llm_cache: bool = os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"
    addon_params: Dict[str, Any] = field(default_factory=dict)  # Additional parameters
    tiktoken_model_name: str = os.getenv("TIKTOKEN_MODEL_NAME", "gpt-4")  # For token counting
    
    # Experimental features
    enable_entity_extraction: bool = True
    enable_relationship_building: bool = True
    enable_graph_optimization: bool = False
    
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure directories exist"""
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "RAGEngineConfig":
        """Create configuration from environment variables"""
        return cls(
            embedding=EmbeddingConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            ),
            llm=LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            ),
            vision=VisionConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
        )
