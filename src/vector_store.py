"""
Vektör Veritabanı Modülü
ChromaDB ile vektör saklama ve arama işlemleri.
LangChain Chroma wrapper kullanır.
"""

import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vektör veritabanı yöneticisi - LangChain Chroma wrapper"""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "pdf_collection",
        embeddings: Optional[Embeddings] = None
    ):
        """
        Args:
            persist_directory: ChromaDB kalıcı depolama dizini
            collection_name: Collection adı
            embeddings: LangChain Embeddings objesi (opsiyonel, sonra set edilebilir)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embeddings = embeddings
        
        # Dizini oluştur
        os.makedirs(persist_directory, exist_ok=True)
        
        # LangChain Chroma wrapper'ı başlat
        if embeddings:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=embeddings
            )
        else:
            # Embeddings sonra set edilecek
            self.vectorstore = None
        
        logger.info(f"VectorStore başlatıldı: {collection_name}")
    
    def set_embeddings(self, embeddings: Embeddings):
        """Embeddings'i set et (lazy initialization için)"""
        self._embeddings = embeddings
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=embeddings
        )
        logger.info("Embeddings set edildi")
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ):
        """
        LangChain Document objelerini vektör veritabanına ekler.
        
        Args:
            documents: LangChain Document listesi
            ids: Özel ID'ler (opsiyonel)
        """
        if not documents:
            raise ValueError("Documents boş olamaz")
        
        if not self.vectorstore:
            raise ValueError("Embeddings set edilmemiş. set_embeddings() çağırın.")
        
        logger.info(f"{len(documents)} doküman ekleniyor...")
        
        # LangChain Chroma'nın add_documents metodunu kullan
        if ids:
            self.vectorstore.add_documents(documents=documents, ids=ids)
        else:
            self.vectorstore.add_documents(documents=documents)
        
        logger.info(f"{len(documents)} doküman eklendi")
    
    def add_texts_with_metadata(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Backward compatibility için - eski API'yi destekler.
        Dokümanları vektör veritabanına ekler.
        
        Args:
            texts: Metin listesi
            embeddings: Embedding vektörleri (kullanılmaz, LangChain otomatik oluşturur)
            metadatas: Metadata listesi
            ids: Özel ID'ler (opsiyonel)
        """
        if not texts or not metadatas:
            raise ValueError("Texts ve metadatas boş olamaz")
        
        if len(texts) != len(metadatas):
            raise ValueError("Texts ve metadatas aynı uzunlukta olmalı")
        
        # Document objelerine dönüştür
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc_id = ids[i] if ids and i < len(ids) else None
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        self.add_documents(documents, ids=ids)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        LangChain'in similarity_search_with_score metodunu kullanır.
        
        Args:
            query: Sorgu metni
            k: Döndürülecek en iyi sonuç sayısı
            filter: Metadata filtresi (opsiyonel)
            
        Returns:
            List[tuple]: (Document, score) tuple listesi
        """
        if not self.vectorstore:
            raise ValueError("Embeddings set edilmemiş. set_embeddings() çağırın.")
        
        if filter:
            return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)
        else:
            return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Backward compatibility için - eski API'yi destekler.
        Vektör veritabanında arama yapar.
        
        Args:
            query_embedding: Sorgu embedding vektörü (kullanılmaz, query string gerekli)
            top_k: Döndürülecek en iyi sonuç sayısı
            filter_metadata: Metadata filtresi (opsiyonel)
            
        Returns:
            List[Dict]: Her sonuç için {'text': str, 'metadata': dict, 'distance': float}
        """
        # Bu metod artık kullanılmamalı, similarity_search_with_score kullanılmalı
        # Ama backward compatibility için korunuyor
        raise NotImplementedError(
            "Bu metod artık desteklenmiyor. "
            "query string ile similarity_search_with_score() kullanın."
        )
    
    def delete_collection(self):
        """Collection'ı siler"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection silindi: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection silinirken hata: {e}")
    
    def get_collection_info(self) -> Dict:
        """Collection hakkında bilgi döndürür"""
        if not self.vectorstore:
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'persist_directory': self.persist_directory
            }
        
        # Chroma collection'a eriş
        collection = self.vectorstore._collection
        count = collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'persist_directory': self.persist_directory
        }
    
    def reset_collection(self):
        """Collection'ı sıfırlar (tüm dokümanları siler)"""
        self.delete_collection()
        if self._embeddings:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self._embeddings
            )
        logger.info("Collection sıfırlandı")
    
    def delete_collection(self):
        """Collection'ı siler"""
        try:
            if self.vectorstore:
                # Chroma collection'ı sil
                self.vectorstore.delete_collection()
            logger.info(f"Collection silindi: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection silinirken hata: {e}")
    
    def list_collections(self) -> List[Dict]:
        """Tüm collection'ları listeler"""
        try:
            from langchain_chroma import Chroma
            import chromadb
            
            # ChromaDB client oluştur
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = client.list_collections()
            
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
        if self._embeddings:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                collection_name=collection_name,
                embedding_function=self._embeddings
            )
        logger.info(f"Collection değiştirildi: {collection_name}")
    
    def as_retriever(self, **kwargs):
        """
        LangChain retriever oluşturur (Chains için gerekli).
        
        Args:
            **kwargs: Retriever parametreleri (search_kwargs, vb.)
            
        Returns:
            VectorStoreRetriever: LangChain retriever objesi
        """
        if not self.vectorstore:
            raise ValueError("Embeddings set edilmemiş. set_embeddings() çağırın.")
        
        return self.vectorstore.as_retriever(**kwargs)

