#!/usr/bin/env python3
"""
RAG PDF Chatbot - Ana CLI Uygulaması
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
import logging

from src.pdf_processor import PDFProcessor
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_handler import OllamaLLMHandler
from src.rag_chain import RAGChain
from src.memory import MemoryManager
from langchain.schema import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables yükle
load_dotenv()


class RAGChatbot:
    """RAG PDF Chatbot ana sınıfı"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Yapılandırma dosyasını yükler ve bileşenleri başlatır"""
        self.config = self._load_config(config_path)
        self._initialize_components()
        self.current_pdf = None
    
    def _load_config(self, config_path: str) -> dict:
        """Yapılandırma dosyasını yükler"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Yapılandırma yüklendi: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Yapılandırma dosyası bulunamadı: {config_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Yapılandırma yüklenirken hata: {e}")
            sys.exit(1)
    
    def _initialize_components(self):
        """Tüm bileşenleri başlatır"""
        # PDF Processor
        pdf_config = self.config.get('pdf', {})
        self.pdf_processor = PDFProcessor(
            max_file_size_mb=pdf_config.get('max_file_size_mb', 50)
        )
        
        # Text Splitter
        chunking_config = self.config.get('chunking', {})
        self.text_splitter = TextSplitter(
            chunk_size=chunking_config.get('chunk_size', 500),
            chunk_overlap=chunking_config.get('chunk_overlap', 150),
            separators=chunking_config.get('separators', ["\n\n", "\n", ". ", " ", ""])
        )
        
        # Embedding Generator
        embedding_config = self.config.get('embedding', {})
        model_name = os.getenv('EMBEDDING_MODEL') or embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_generator = EmbeddingGenerator(model_name=model_name)
        
        # Vector Store (LangChain Chroma)
        vector_db_config = self.config.get('vector_db', {})
        persist_dir = os.getenv('CHROMA_PERSIST_DIRECTORY') or vector_db_config.get('persist_directory', './data/chroma_db')
        collection_name = vector_db_config.get('collection_name_prefix', 'pdf_collection')
        self.vector_store = VectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embeddings=self.embedding_generator.get_langchain_embeddings()
        )
        
        # LLM Handler (LangChain Ollama)
        llm_config = self.config.get('llm', {})
        base_url = os.getenv('OLLAMA_BASE_URL') or llm_config.get('base_url', 'http://localhost:11434')
        model_name = os.getenv('OLLAMA_MODEL') or llm_config.get('model_name', 'mistral')
        use_chat = llm_config.get('use_chat', False)
        self.llm_handler = OllamaLLMHandler(
            model_name=model_name,
            base_url=base_url,
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 1000),
            timeout=llm_config.get('timeout', 30),
            use_chat=use_chat
        )
        
        # Memory Manager (LangChain Memory)
        memory_config = self.config.get('memory', {})
        memory_enabled = memory_config.get('enabled', True)
        if memory_enabled:
            memory_type = memory_config.get('type', 'buffer')
            window_size = memory_config.get('window_size', 10)
            self.memory_manager = MemoryManager(
                memory_type=memory_type,
                window_size=window_size,
                llm=self.llm_handler.get_langchain_llm() if memory_type == 'summary' else None
            )
        else:
            self.memory_manager = None
        
        # RAG Chain (LangChain Chains)
        rag_config = self.config.get('rag', {})
        chain_type = rag_config.get('chain_type', 'stuff')
        return_source_documents = rag_config.get('return_source_documents', True)
        self.rag_chain = RAGChain(
            vector_store=self.vector_store,
            llm_handler=self.llm_handler,
            memory_manager=self.memory_manager,
            chain_type=chain_type,
            top_k=rag_config.get('top_k', 5),
            return_source_documents=return_source_documents
        )
        
        self.current_collection_name = None  # Aktif collection adı
        logger.info("Tüm bileşenler başlatıldı (LangChain entegrasyonu ile)")
    
    def load_pdf(self, pdf_path: str):
        """PDF dosyasını yükler ve işler"""
        try:
            logger.info(f"PDF yükleniyor: {pdf_path}")
            
            # PDF bilgilerini al
            pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
            self.current_pdf = pdf_info['filename']
            
            print(f"\n{'='*50}")
            print(f"PDF Yükleniyor: {pdf_info['filename']}")
            print(f"Sayfa Sayısı: {pdf_info['total_pages']}")
            print(f"Dosya Boyutu: {pdf_info['file_size_mb']} MB")
            print(f"{'='*50}\n")
            
            # LangChain Document Loader ile PDF yükle
            print("PDF LangChain Document Loader ile yükleniyor...")
            documents = self.pdf_processor.load_documents(pdf_path)
            print(f"✓ {len(documents)} sayfa yüklendi\n")
            
            # Chunk'lara böl (LangChain TextSplitter ile)
            print("Metin chunk'lara bölünüyor...")
            chunking_config = self.config.get('chunking', {})
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.get('chunk_size', 500),
                chunk_overlap=chunking_config.get('chunk_overlap', 150),
                separators=chunking_config.get('separators', ["\n\n", "\n", ". ", " ", ""])
            )
            
            # Document'leri chunk'lara böl
            chunked_documents = text_splitter.split_documents(documents)
            print(f"✓ {len(chunked_documents)} chunk oluşturuldu\n")
            
            # Collection'ı PDF dosya adına göre oluştur/güncelle
            collection_name = f"pdf_{Path(pdf_info['filename']).stem}"
            self.vector_store.switch_collection(collection_name)
            self.current_collection_name = collection_name  # Aktif collection'ı kaydet
            
            # Vektör DB'ye kaydet (LangChain Chroma ile)
            print("Vektör veritabanına kaydediliyor...")
            ids = [f"{pdf_info['filename']}_chunk_{i+1}" for i in range(len(chunked_documents))]
            self.vector_store.add_documents(chunked_documents, ids=ids)
            print(f"✓ {len(chunked_documents)} doküman kaydedildi\n")
            
            # RAG chain'i yeniden oluştur (yeni collection için)
            rag_config = self.config.get('rag', {})
            chain_type = rag_config.get('chain_type', 'stuff')
            return_source_documents = rag_config.get('return_source_documents', True)
            self.rag_chain = RAGChain(
                vector_store=self.vector_store,
                llm_handler=self.llm_handler,
                memory_manager=self.memory_manager,
                chain_type=chain_type,
                top_k=rag_config.get('top_k', 5),
                return_source_documents=return_source_documents
            )
            
            print(f"{'='*50}")
            print(f"✓ PDF başarıyla yüklendi ve işlendi!")
            print(f"✓ Collection: {collection_name}")
            print(f"{'='*50}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"PDF yükleme hatası: {e}")
            print(f"\n❌ Hata: {e}\n")
            return False
    
    def chat(self):
        """Sohbet modunu başlatır"""
        # Aktif collection'ı belirle
        if self.current_collection_name:
            # Son yüklenen PDF'in collection'ını kullan
            self.vector_store.switch_collection(self.current_collection_name)
            # RAG chain'i yeniden oluştur
            self._recreate_rag_chain()
            print(f"\n✓ Collection yüklendi: {self.current_collection_name}\n")
        else:
            # Mevcut collection'ları kontrol et
            collections = self.vector_store.list_collections()
            collections_with_docs = [c for c in collections if c['count'] > 0]
            
            if not collections_with_docs:
                print("\n❌ Henüz PDF yüklenmemiş. Önce bir PDF yükleyin.\n")
                return
            
            # Eğer tek bir collection varsa onu kullan
            if len(collections_with_docs) == 1:
                collection_name = collections_with_docs[0]['name']
                self.vector_store.switch_collection(collection_name)
                self.current_collection_name = collection_name
                self._recreate_rag_chain()
                print(f"\n✓ Collection yüklendi: {collection_name} ({collections_with_docs[0]['count']} doküman)\n")
            else:
                # Birden fazla collection varsa kullanıcıya seçtir
                print("\nMevcut PDF collection'ları:")
                for i, col in enumerate(collections_with_docs, 1):
                    print(f"  [{i}] {col['name']} ({col['count']} doküman)")
                
                while True:
                    try:
                        choice = input("\nKullanmak istediğiniz collection numarasını girin: ").strip()
                        idx = int(choice) - 1
                        if 0 <= idx < len(collections_with_docs):
                            collection_name = collections_with_docs[idx]['name']
                            self.vector_store.switch_collection(collection_name)
                            self.current_collection_name = collection_name
                            self._recreate_rag_chain()
                            print(f"\n✓ Collection yüklendi: {collection_name}\n")
                            break
                        else:
                            print("❌ Geçersiz seçim. Lütfen geçerli bir numara girin.")
                    except ValueError:
                        print("❌ Lütfen bir sayı girin.")
                    except KeyboardInterrupt:
                        print("\n\nİşlem iptal edildi.\n")
                        return
        
        # Memory temizle (yeni sohbet başlatılıyor)
        if self.memory_manager:
            self.memory_manager.clear()
        
        print("\n" + "="*50)
        print("RAG PDF Sohbet Botu - Sohbet Modu (LangChain)")
        if self.memory_manager:
            print(f"Memory: {self.memory_manager.memory_type}")
        print("="*50)
        print("Çıkmak için '/exit' veya '/quit' yazın")
        print("Memory'yi temizlemek için '/clear' yazın\n")
        
        while True:
            try:
                question = input("Siz: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['/exit', '/quit', '/çıkış']:
                    print("\nSohbet sonlandırıldı.\n")
                    break
                
                if question.lower() == '/clear':
                    if self.memory_manager:
                        self.memory_manager.clear()
                        print("\n✓ Memory temizlendi\n")
                    continue
                
                # RAG sorgusu (LangChain chain ile)
                result = self.rag_chain.query(question)
                
                # Cevabı göster
                print(f"\nBot: {result['answer']}\n")
                
                # Kaynakları göster
                if result['sources'] and self.config.get('cli', {}).get('show_sources', True):
                    print("Kaynaklar:")
                    for source in result['sources']:
                        similarity_str = f" (Benzerlik: {source['similarity']})" if source['similarity'] else ""
                        print(f"  - {source['source_file']}, Sayfa {source['page']}{similarity_str}")
                    print()
                
            except KeyboardInterrupt:
                print("\n\nSohbet sonlandırıldı.\n")
                break
            except Exception as e:
                logger.error(f"Sohbet hatası: {e}")
                print(f"\n❌ Hata: {e}\n")
    
    def _recreate_rag_chain(self):
        """RAG chain'i yeniden oluşturur (collection değiştiğinde)"""
        rag_config = self.config.get('rag', {})
        chain_type = rag_config.get('chain_type', 'stuff')
        return_source_documents = rag_config.get('return_source_documents', True)
        self.rag_chain = RAGChain(
            vector_store=self.vector_store,
            llm_handler=self.llm_handler,
            memory_manager=self.memory_manager,
            chain_type=chain_type,
            top_k=rag_config.get('top_k', 5),
            return_source_documents=return_source_documents
        )


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='RAG PDF Chatbot')
    parser.add_argument(
        'command',
        nargs='?',
        choices=['load', 'chat'],
        help='Komut: load (PDF yükle) veya chat (sohbet başlat)'
    )
    parser.add_argument(
        'file',
        nargs='?',
        help='PDF dosya yolu (load komutu için)'
    )
    parser.add_argument(
        '--chat',
        action='store_true',
        help='PDF yüklendikten sonra sohbet modunu başlat'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Yapılandırma dosyası yolu (varsayılan: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Chatbot'u başlat
    chatbot = RAGChatbot(config_path=args.config)
    
    # Komut işle
    if args.command == 'load':
        if not args.file:
            print("❌ PDF dosya yolu belirtilmedi.")
            sys.exit(1)
        
        success = chatbot.load_pdf(args.file)
        if success and args.chat:
            chatbot.chat()
    
    elif args.command == 'chat':
        chatbot.chat()
    
    else:
        # İnteraktif menü
        while True:
            print("\n" + "="*50)
            print("=== RAG PDF Sohbet Botu ===")
            print("="*50)
            print("[1] PDF Yükle")
            print("[2] Sohbet Başlat")
            print("[3] Çıkış")
            print("="*50)
            
            choice = input("\nSeçiminiz: ").strip()
            
            if choice == '1':
                pdf_path = input("PDF dosya yolu: ").strip()
                if pdf_path:
                    chatbot.load_pdf(pdf_path)
            
            elif choice == '2':
                chatbot.chat()
            
            elif choice == '3':
                print("\nÇıkılıyor...\n")
                break
            
            else:
                print("\n❌ Geçersiz seçim. Lütfen 1, 2 veya 3 girin.\n")


if __name__ == "__main__":
    main()

