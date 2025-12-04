"""
Embedding Modülü
Metinleri vektörlere dönüştürür.
LangChain HuggingFaceEmbeddings kullanır.
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Embedding oluşturan sınıf - LangChain HuggingFaceEmbeddings wrapper"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformers model adı
        """
        self.model_name = model_name
        logger.info(f"Embedding modeli yükleniyor: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding modeli yüklendi")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Tek bir metin için embedding oluşturur.
        
        Args:
            text: Embedding oluşturulacak metin
            
        Returns:
            List[float]: Embedding vektörü
        """
        if not text or not text.strip():
            raise ValueError("Boş metin için embedding oluşturulamaz")
        
        embedding = self.embeddings.embed_query(text)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Birden fazla metin için batch embedding oluşturur.
        
        Args:
            texts: Embedding oluşturulacak metin listesi
            show_progress: İlerleme çubuğu göster (LangChain'de desteklenmiyor, parametre korunuyor)
            
        Returns:
            List[List[float]]: Embedding vektörleri listesi
        """
        if not texts:
            return []
        
        # Boş metinleri filtrele
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("Geçerli metin bulunamadı")
        
        logger.info(f"{len(valid_texts)} metin için embedding oluşturuluyor...")
        
        # LangChain'in embed_documents metodunu kullan
        embeddings = self.embeddings.embed_documents(valid_texts)
        
        logger.info(f"{len(embeddings)} embedding oluşturuldu")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Embedding vektörünün boyutunu döndürür.
        
        Returns:
            int: Vektör boyutu
        """
        # Test embedding ile boyutu öğren
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
    
    def get_langchain_embeddings(self) -> Embeddings:
        """
        LangChain Embeddings objesini döndürür (VectorStore için).
        
        Returns:
            Embeddings: LangChain Embeddings objesi
        """
        return self.embeddings

