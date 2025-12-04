"""
RAG Chain Modülü
RAG pipeline'ını yönetir.
"""

from typing import List, Dict, Optional
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .llm_handler import OllamaLLMHandler
from .prompt_templates import PromptTemplates
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChain:
    """RAG pipeline yöneticisi"""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        llm_handler: OllamaLLMHandler,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ):
        """
        Args:
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
            llm_handler: LLM handler instance
            top_k: Top-K benzer chunk sayısı
            similarity_threshold: Minimum benzerlik skoru
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.prompt_templates = PromptTemplates()
    
    def query(self, question: str, filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Kullanıcı sorusuna RAG ile cevap verir.
        
        Args:
            question: Kullanıcı sorusu
            filter_metadata: Metadata filtresi (opsiyonel)
            
        Returns:
            Dict: {'answer': str, 'sources': List[Dict]}
        """
        logger.info(f"Soru işleniyor: {question[:50]}...")
        
        # 1. Soruyu embedding'e dönüştür
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # 2. Vektör DB'de arama yap
        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filter_metadata=filter_metadata
        )
        
        if not retrieved_chunks:
            return {
                'answer': 'İlgili bilgi bulunamadı.',
                'sources': []
            }
        
        # 3. Benzerlik skoruna göre filtrele
        filtered_chunks = []
        for chunk in retrieved_chunks:
            distance = chunk.get('distance', 1.0)
            similarity = 1 - distance  # Cosine distance -> similarity
            if similarity >= self.similarity_threshold:
                filtered_chunks.append(chunk)
        
        if not filtered_chunks:
            return {
                'answer': 'İlgili bilgi bulunamadı (benzerlik eşiğinin altında).',
                'sources': []
            }
        
        # 4. Context oluştur
        context = self.prompt_templates.format_context(filtered_chunks)
        
        # 5. Prompt hazırla
        prompt = self.prompt_templates.rag_prompt(context=context, question=question)
        
        # 6. LLM'e gönder
        try:
            answer = self.llm_handler.generate(prompt, stream=False)
        except Exception as e:
            logger.error(f"LLM hatası: {e}")
            answer = "Üzgünüm, cevap oluşturulurken bir hata oluştu."
        
        # 7. Kaynakları formatla
        sources = self.prompt_templates.format_sources(filtered_chunks)
        
        return {
            'answer': answer.strip(),
            'sources': sources
        }
    
    def query_streaming(self, question: str, filter_metadata: Optional[Dict] = None):
        """
        Streaming modda RAG sorgusu (gelecekte kullanılabilir).
        
        Args:
            question: Kullanıcı sorusu
            filter_metadata: Metadata filtresi
            
        Yields:
            str: Streaming cevap parçaları
        """
        # 1-4. Aynı işlemler
        query_embedding = self.embedding_generator.generate_embedding(question)
        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filter_metadata=filter_metadata
        )
        
        if not retrieved_chunks:
            yield "İlgili bilgi bulunamadı."
            return
        
        context = self.prompt_templates.format_context(retrieved_chunks)
        prompt = self.prompt_templates.rag_prompt(context=context, question=question)
        
        # 5. Streaming response
        try:
            for chunk in self.llm_handler.generate(prompt, stream=True):
                yield chunk
        except Exception as e:
            logger.error(f"LLM streaming hatası: {e}")
            yield "Üzgünüm, cevap oluşturulurken bir hata oluştu."

