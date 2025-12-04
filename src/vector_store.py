"""
Vektör Veritabanı Modülü
ChromaDB ile vektör saklama ve arama işlemleri.
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vektör veritabanı yöneticisi"""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "pdf_collection"
    ):
        """
        Args:
            persist_directory: ChromaDB kalıcı depolama dizini
            collection_name: Collection adı
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Dizini oluştur
        os.makedirs(persist_directory, exist_ok=True)
        
        # ChromaDB client oluştur
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collection'ı al veya oluştur
        self.collection = self._get_or_create_collection(collection_name)
        logger.info(f"VectorStore başlatıldı: {collection_name}")
    
    def _get_or_create_collection(self, name: str):
        """Collection'ı al veya oluştur"""
        try:
            collection = self.client.get_collection(name=name)
            logger.info(f"Mevcut collection yüklendi: {name}")
            return collection
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )
            logger.info(f"Yeni collection oluşturuldu: {name}")
            return collection
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Dokümanları vektör veritabanına ekler.
        
        Args:
            texts: Metin listesi
            embeddings: Embedding vektörleri
            metadatas: Metadata listesi
            ids: Özel ID'ler (opsiyonel)
        """
        if not texts or not embeddings or not metadatas:
            raise ValueError("Texts, embeddings ve metadatas boş olamaz")
        
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("Texts, embeddings ve metadatas aynı uzunlukta olmalı")
        
        # ID'leri oluştur
        if ids is None:
            ids = [f"chunk_{i+1}" for i in range(len(texts))]
        
        logger.info(f"{len(texts)} doküman ekleniyor...")
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"{len(texts)} doküman eklendi")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Vektör veritabanında arama yapar.
        
        Args:
            query_embedding: Sorgu embedding vektörü
            top_k: Döndürülecek en iyi sonuç sayısı
            filter_metadata: Metadata filtresi (opsiyonel)
            
        Returns:
            List[Dict]: Her sonuç için {'text': str, 'metadata': dict, 'distance': float}
        """
        where_clause = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        # Sonuçları formatla
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i] if results['ids'] else None
                })
        
        return formatted_results
    
    def delete_collection(self):
        """Collection'ı siler"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection silindi: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection silinirken hata: {e}")
    
    def get_collection_info(self) -> Dict:
        """Collection hakkında bilgi döndürür"""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'persist_directory': self.persist_directory
        }
    
    def reset_collection(self):
        """Collection'ı sıfırlar (tüm dokümanları siler)"""
        self.delete_collection()
        self.collection = self._get_or_create_collection(self.collection_name)
        logger.info("Collection sıfırlandı")
    
    def list_collections(self) -> List[Dict]:
        """Tüm collection'ları listeler"""
        try:
            collections = self.client.list_collections()
            result = []
            for col in collections:
                try:
                    count = col.count()
                    result.append({
                        'name': col.name,
                        'count': count
                    })
                except Exception as e:
                    logger.warning(f"Collection {col.name} bilgisi alınamadı: {e}")
            return result
        except Exception as e:
            logger.error(f"Collection listesi alınamadı: {e}")
            return []
    
    def switch_collection(self, collection_name: str):
        """Aktif collection'ı değiştirir"""
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)
        logger.info(f"Collection değiştirildi: {collection_name}")

