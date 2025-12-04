"""
RAG Chain Modülü
RAG pipeline'ını yönetir.
LangChain Chains kullanır.
"""

from typing import List, Dict, Optional, Iterator
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.schema import Document
from .vector_store import VectorStore
from .llm_handler import OllamaLLMHandler
from .prompt_templates import PromptTemplates
from .memory import MemoryManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChain:
    """RAG pipeline yöneticisi - LangChain Chains kullanır"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_handler: OllamaLLMHandler,
        memory_manager: Optional[MemoryManager] = None,
        chain_type: str = "stuff",
        top_k: int = 5,
        return_source_documents: bool = True
    ):
        """
        Args:
            vector_store: Vector store instance (LangChain Chroma)
            llm_handler: LLM handler instance
            memory_manager: Memory manager (opsiyonel, ConversationalRetrievalChain için)
            chain_type: Chain tipi ("stuff", "map_reduce", "refine", "map_rerank")
            top_k: Top-K benzer chunk sayısı
            return_source_documents: Kaynak dokümanları döndür
        """
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.memory_manager = memory_manager
        self.chain_type = chain_type
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.prompt_templates = PromptTemplates()
        
        # LangChain chain'i oluştur
        self.chain = self._create_chain()
    
    def _create_chain(self) -> BaseRetrievalQA:
        """LangChain chain'ini oluşturur"""
        llm = self.llm_handler.get_langchain_llm()
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        if self.memory_manager:
            # ConversationalRetrievalChain kullan (memory ile)
            memory = self.memory_manager.get_memory()
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=self.return_source_documents,
                verbose=True
            )
            logger.info("ConversationalRetrievalChain oluşturuldu")
        else:
            # RetrievalQA kullan (basit RAG)
            prompt_template = self.prompt_templates.get_rag_template()
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=self.chain_type,
                retriever=retriever,
                return_source_documents=self.return_source_documents,
                chain_type_kwargs={"prompt": prompt_template},
                verbose=True
            )
            logger.info(f"RetrievalQA chain oluşturuldu (type: {self.chain_type})")
        
        return chain
    
    def query(self, question: str, filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Kullanıcı sorusuna RAG ile cevap verir.
        
        Args:
            question: Kullanıcı sorusu
            filter_metadata: Metadata filtresi (opsiyonel, retriever'da kullanılabilir)
            
        Returns:
            Dict: {'answer': str, 'sources': List[Dict]}
        """
        logger.info(f"Soru işleniyor: {question[:50]}...")
        
        try:
            # LangChain chain'i kullan
            if isinstance(self.chain, ConversationalRetrievalChain):
                # ConversationalRetrievalChain için
                result = self.chain({"question": question})
                answer = result.get("answer", "")
                source_documents = result.get("source_documents", [])
            else:
                # RetrievalQA için
                result = self.chain({"query": question})
                answer = result.get("result", "")
                source_documents = result.get("source_documents", [])
            
            # Kaynakları formatla
            sources = self.prompt_templates.format_sources(
                chunks=[],
                documents=source_documents
            )
            
            return {
                'answer': answer.strip() if answer else "Cevap oluşturulamadı.",
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"RAG chain hatası: {e}")
            return {
                'answer': "Üzgünüm, cevap oluşturulurken bir hata oluştu.",
                'sources': []
            }
    
    def query_streaming(self, question: str, filter_metadata: Optional[Dict] = None) -> Iterator[str]:
        """
        Streaming modda RAG sorgusu.
        
        Args:
            question: Kullanıcı sorusu
            filter_metadata: Metadata filtresi (opsiyonel)
            
        Yields:
            str: Streaming cevap parçaları
        """
        try:
            # LangChain chain streaming
            if isinstance(self.chain, ConversationalRetrievalChain):
                for chunk in self.chain.stream({"question": question}):
                    if "answer" in chunk:
                        yield chunk["answer"]
            else:
                # RetrievalQA streaming desteği sınırlı
                result = self.chain({"query": question})
                answer = result.get("result", "")
                # Basit streaming simülasyonu
                words = answer.split()
                for word in words:
                    yield word + " "
        except Exception as e:
            logger.error(f"RAG streaming hatası: {e}")
            yield "Üzgünüm, cevap oluşturulurken bir hata oluştu."

