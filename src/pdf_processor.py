"""
PDF İşleme Modülü
PDF dosyalarını yükler ve metin içeriğini çıkarır.
LangChain Document Loader kullanır.
"""

import os
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF dosyalarını işleyen sınıf"""
    
    def __init__(self, max_file_size_mb: int = 50):
        """
        Args:
            max_file_size_mb: Maksimum dosya boyutu (MB)
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate_pdf(self, file_path: str) -> bool:
        """
        PDF dosyasının geçerliliğini kontrol eder.
        
        Args:
            file_path: PDF dosya yolu
            
        Returns:
            bool: Dosya geçerliyse True
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Sadece .pdf dosyaları desteklenmektedir")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"Dosya boyutu çok büyük: {file_size / 1024 / 1024:.2f} MB "
                f"(Maksimum: {self.max_file_size_mb} MB)"
            )
        
        return True
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        PDF'den LangChain Document objelerini yükler.
        
        Args:
            file_path: PDF dosya yolu
            
        Returns:
            List[Document]: LangChain Document objeleri (sayfa numarası metadata ile)
        """
        self.validate_pdf(file_path)
        
        try:
            # LangChain PyPDFLoader kullan
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Sayfa numarası metadata'sını ekle
            filename = os.path.basename(file_path)
            for doc in documents:
                if 'page' not in doc.metadata:
                    # PyPDFLoader bazen page metadata eklemiyor, ekleyelim
                    doc.metadata['source_file'] = filename
                    doc.metadata['file_path'] = file_path
                else:
                    doc.metadata['source_file'] = filename
                    doc.metadata['file_path'] = file_path
            
            logger.info(f"PDF yüklendi: {len(documents)} sayfa")
            return documents
            
        except Exception as e:
            logger.error(f"PDF işleme hatası: {e}")
            raise
    
    def extract_text(self, file_path: str) -> List[Dict[str, any]]:
        """
        PDF'den metin içeriğini çıkarır ve sayfa bazlı döndürür.
        (Backward compatibility için korunuyor)
        
        Args:
            file_path: PDF dosya yolu
            
        Returns:
            List[Dict]: Her sayfa için {'page': int, 'text': str} formatında liste
        """
        documents = self.load_documents(file_path)
        
        pages_content = []
        for doc in documents:
            page_num = doc.metadata.get('page', 0)
            if doc.page_content and doc.page_content.strip():
                pages_content.append({
                    'page': page_num,
                    'text': doc.page_content.strip()
                })
        
        return pages_content
    
    def get_pdf_info(self, file_path: str) -> Dict[str, any]:
        """
        PDF dosyası hakkında bilgi döndürür.
        
        Args:
            file_path: PDF dosya yolu
            
        Returns:
            Dict: PDF bilgileri
        """
        self.validate_pdf(file_path)
        
        try:
            # LangChain loader ile sayfa sayısını al
            documents = self.load_documents(file_path)
            total_pages = len(documents)
            file_size = os.path.getsize(file_path)
            
            return {
                'filename': os.path.basename(file_path),
                'file_path': file_path,
                'total_pages': total_pages,
                'file_size_mb': round(file_size / 1024 / 1024, 2)
            }
        except Exception as e:
            logger.error(f"PDF bilgisi alınamadı: {e}")
            raise

