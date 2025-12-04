"""
Conversation Memory Modülü
LangChain Memory yapılarını yönetir.
"""

from typing import Optional, Any
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryManager:
    """Conversation memory yöneticisi"""
    
    def __init__(
        self,
        memory_type: str = "buffer",
        window_size: int = 10,
        llm=None
    ):
        """
        Args:
            memory_type: Memory tipi ("buffer", "window", "summary")
            window_size: Window memory için pencere boyutu
            llm: Summary memory için LLM (opsiyonel)
        """
        self.memory_type = memory_type
        self.window_size = window_size
        self.llm = llm
        self.memory = self._create_memory()
        logger.info(f"Memory başlatıldı: {memory_type}")
    
    def _create_memory(self) -> Any:
        """Memory objesini oluşturur"""
        if self.memory_type == "buffer":
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif self.memory_type == "window":
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=self.window_size
            )
        elif self.memory_type == "summary":
            if not self.llm:
                logger.warning("Summary memory için LLM gerekli, buffer memory kullanılıyor")
                return ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        else:
            logger.warning(f"Bilinmeyen memory tipi: {self.memory_type}, buffer kullanılıyor")
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
    
    def get_memory(self) -> Any:
        """
        LangChain Memory objesini döndürür (Chains için).
        
        Returns:
            Any: LangChain Memory objesi
        """
        return self.memory
    
    def clear(self):
        """Memory'yi temizler"""
        self.memory.clear()
        logger.info("Memory temizlendi")
    
    def save_context(self, input_str: str, output_str: str):
        """
        Context'i memory'ye kaydeder.
        
        Args:
            input_str: Kullanıcı girdisi
            output_str: Model çıktısı
        """
        self.memory.save_context({"input": input_str}, {"output": output_str})
    
    def load_memory_variables(self, inputs: dict) -> dict:
        """
        Memory değişkenlerini yükler.
        
        Args:
            inputs: Input değişkenleri
            
        Returns:
            dict: Memory değişkenleri
        """
        return self.memory.load_memory_variables(inputs)

