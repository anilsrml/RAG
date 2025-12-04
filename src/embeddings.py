"""
Embedding Modülü
Metinleri vektörlere dönüştürür.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Embedding oluşturan sınıf"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformers model adı
        """
        self.model_name = model_name
        logger.info(f"Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
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
        
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Birden fazla metin için batch embedding oluşturur.
        
        Args:
            texts: Embedding oluşturulacak metin listesi
            show_progress: İlerleme çubuğu göster
            
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
        
        embeddings = self.model.encode(
            valid_texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=32
        )
        
        logger.info(f"{len(embeddings)} embedding oluşturuldu")
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Embedding vektörünün boyutunu döndürür.
        
        Returns:
            int: Vektör boyutu
        """
        # Test embedding ile boyutu öğren
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)

