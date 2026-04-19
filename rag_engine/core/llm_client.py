"""
LLM Client for knowledge graph extraction

Provides a unified interface for calling LLM models (OpenAI, Ollama, etc.)
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from rag_engine.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for entity and relationship extraction"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client
        
        Args:
            config: LLMConfig instance with model settings
        """
        self.config = config or LLMConfig()
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate LLM client based on config"""
        import os
        
        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        
        # For OpenAI models
        if "gpt" in self.config.model.lower():
            if not api_key:
                logger.warning(f"OpenAI API key not found for model {self.config.model}")
                logger.warning("Using mock LLM client. For real extraction, set OPENAI_API_KEY environment variable")
                self.client = None
                self.client_type = "mock"
                return
            
            try:
                from openai import AsyncOpenAI
                
                self.client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url
                )
                self.client_type = "openai"
                logger.info(f"Initialized OpenAI client for model: {self.config.model}")
            except ImportError:
                logger.error("OpenAI package not installed. Install with: pip install openai")
                self.client = None
        
        # For Ollama models
        elif "ollama" in self.config.model.lower() or (self.config.base_url and "ollama" in self.config.base_url):
            try:
                from openai import AsyncOpenAI
                
                base_url = self.config.base_url or "http://localhost:11434/v1"
                self.client = AsyncOpenAI(
                    api_key="ollama",  # Dummy key for Ollama
                    base_url=base_url
                )
                self.client_type = "ollama"
                logger.info(f"Initialized Ollama client for model: {self.config.model}")
            except ImportError:
                logger.error("OpenAI package not installed")
                self.client = None
        
        # Default to OpenAI-compatible client
        else:
            if not api_key:
                logger.warning(f"No API key found for model {self.config.model}")
                logger.warning("Using mock LLM client. For real extraction, set API key or base_url")
                self.client = None
                self.client_type = "mock"
                return
            
            try:
                from openai import AsyncOpenAI
                
                self.client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url
                )
                self.client_type = "openai_compatible"
                logger.info(f"Initialized OpenAI-compatible client for model: {self.config.model}")
            except ImportError:
                logger.error("OpenAI package not installed")
                self.client = None
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate completion from LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Returns:
            Generated text response
        """
        if self.client is None or self.client_type == "mock":
            logger.debug(f"LLM client not initialized ({self.client_type}). Returning empty extraction.")
            # Return empty extraction with completion delimiter for parsing
            return "<|COMPLETE|>"
        
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.model_kwargs
            )
            
            # Extract response text
            result = response.choices[0].message.content
            logger.debug(f"LLM generated {len(result)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "<|COMPLETE|>"
    
    async def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate with explicit system and user prompts
        
        Args:
            system_prompt: System message
            user_prompt: User message
        
        Returns:
            Generated text response
        """
        return await self.generate(user_prompt, system_prompt)


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """
    Get or create global LLM client
    
    Args:
        config: Optional LLMConfig for initialization
    
    Returns:
        LLMClient instance
    """
    global _llm_client
    
    if _llm_client is None:
        _llm_client = LLMClient(config)
    
    return _llm_client


def set_llm_client(client: Optional[LLMClient]) -> None:
    """Set global LLM client"""
    global _llm_client
    _llm_client = client
