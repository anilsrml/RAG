"""
Text Chunking Modülü
Metinleri chunk'lara böler ve metadata ekler.
"""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSplitter:
    """Metinleri chunk'lara bölen sınıf"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 150,
        separators: List[str] = None
    ):
        """
        Args:
            chunk_size: Chunk boyutu (karakter)
            chunk_overlap: Overlap boyutu (karakter)
            separators: Bölme ayırıcıları (öncelik sırasına göre)
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    def split_pages(self, pages_content: List[Dict], source_filename: str) -> List[Dict]:
        """
        Sayfa içeriklerini chunk'lara böler ve metadata ekler.
        
        Args:
            pages_content: PDF processor'dan gelen sayfa içerikleri
            source_filename: Kaynak PDF dosya adı
            
        Returns:
            List[Dict]: Her chunk için {'text': str, 'metadata': dict} formatında liste
        """
        chunks = []
        chunk_id = 0
        
        for page_data in pages_content:
            page_num = page_data['page']
            page_text = page_data['text']
            
            # Sayfa metnini chunk'lara böl
            page_chunks = self.splitter.split_text(page_text)
            
            for chunk_text in page_chunks:
                if chunk_text.strip():  # Boş chunk'ları atla
                    chunk_id += 1
                    chunks.append({
                        'text': chunk_text.strip(),
                        'metadata': {
                            'source_file': source_filename,
                            'page': page_num,
                            'chunk_id': chunk_id,
                            'chunk_size': len(chunk_text)
                        }
                    })
        
        logger.info(f"Toplam {len(chunks)} chunk oluşturuldu")
        return chunks
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Tek bir metni chunk'lara böler.
        
        Args:
            text: Bölünecek metin
            metadata: Chunk'lara eklenecek metadata
            
        Returns:
            List[Dict]: Chunk'lar
        """
        if metadata is None:
            metadata = {}
        
        text_chunks = self.splitter.split_text(text)
        chunks = []
        
        for idx, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = idx + 1
                chunk_metadata['chunk_size'] = len(chunk_text)
                
                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': chunk_metadata
                })
        
        return chunks

