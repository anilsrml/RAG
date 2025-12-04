"""
LLM Handler Modülü
Ollama ile lokal LLM işlemleri.
LangChain Ollama wrapper kullanır.
"""

from typing import Optional, Iterator, Union
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLMHandler:
    """Ollama LLM yöneticisi - LangChain Ollama wrapper"""
    
    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        use_chat: bool = False
    ):
        """
        Args:
            model_name: Ollama model adı
            base_url: Ollama API base URL
            temperature: Model temperature
            max_tokens: Maksimum token sayısı
            timeout: Request timeout (saniye)
            use_chat: ChatOllama kullan (True) veya Ollama kullan (False)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # LangChain Ollama wrapper'ı başlat
        if use_chat:
            self.llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                num_predict=max_tokens,
                timeout=timeout
            )
        else:
            self.llm = Ollama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                num_predict=max_tokens,
                timeout=timeout
            )
        
        logger.info(f"Ollama LLM başlatıldı: {model_name} ({'Chat' if use_chat else 'LLM'})")
    
    def generate(
        self,
        prompt: str,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        LLM'den cevap üretir.
        
        Args:
            prompt: Gönderilecek prompt
            stream: Streaming response isteniyorsa True
            
        Returns:
            str veya Iterator[str]: Cevap metni veya streaming iterator
        """
        try:
            if stream:
                # Streaming response
                return self.llm.stream(prompt)
            else:
                # Normal response
                return self.llm.invoke(prompt)
                
        except Exception as e:
            logger.error(f"LLM çağrısı başarısız: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Ollama bağlantısını test eder"""
        try:
            # Basit bir test çağrısı yap
            test_response = self.llm.invoke("test")
            return test_response is not None
        except Exception:
            return False
    
    def get_langchain_llm(self):
        """
        LangChain LLM objesini döndürür (Chains için).
        
        Returns:
            LLM: LangChain LLM objesi
        """
        return self.llm

