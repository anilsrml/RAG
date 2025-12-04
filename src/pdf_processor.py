"""
PDF İşleme Modülü
PDF dosyalarını yükler ve metin içeriğini çıkarır.
"""

import os
from typing import List, Dict, Optional
import pdfplumber
from pypdf import PdfReader
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
    
    def extract_text(self, file_path: str) -> List[Dict[str, any]]:
        """
        PDF'den metin içeriğini çıkarır ve sayfa bazlı döndürür.
        
        Args:
            file_path: PDF dosya yolu
            
        Returns:
            List[Dict]: Her sayfa için {'page': int, 'text': str} formatında liste
        """
        self.validate_pdf(file_path)
        
        pages_content = []
        
        try:
            # pdfplumber ile daha iyi metin çıkarma
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF açıldı: {total_pages} sayfa")
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            pages_content.append({
                                'page': page_num,
                                'text': text.strip()
                            })
                        else:
                            logger.warning(f"Sayfa {page_num} boş veya metin çıkarılamadı")
                    except Exception as e:
                        logger.error(f"Sayfa {page_num} işlenirken hata: {e}")
                        # Fallback: PyPDF2 ile dene
                        try:
                            reader = PdfReader(file_path)
                            if page_num <= len(reader.pages):
                                text = reader.pages[page_num - 1].extract_text()
                                if text and text.strip():
                                    pages_content.append({
                                        'page': page_num,
                                        'text': text.strip()
                                    })
                        except Exception as fallback_error:
                            logger.error(f"Fallback başarısız (sayfa {page_num}): {fallback_error}")
            
            if not pages_content:
                raise ValueError("PDF'den hiçbir metin çıkarılamadı. Dosya görüntü tabanlı olabilir (OCR gerekebilir).")
            
            logger.info(f"Toplam {len(pages_content)} sayfa işlendi")
            return pages_content
            
        except Exception as e:
            logger.error(f"PDF işleme hatası: {e}")
            raise
    
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
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
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

