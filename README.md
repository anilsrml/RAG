# RAG PDF Chatbot

PDF dokümanlarını yükleyerek, lokalde çalışan bir dil modeli ile doküman içeriği hakkında soru sorabileceğiniz akıllı bir sohbet botu.

## Özellikler

- 📄 PDF dosyalarını yükleme ve işleme
- 🔍 Semantik arama ile ilgili bilgileri bulma
- 💬 Lokal LLM (Ollama) ile sohbet
- 📚 Kaynak gösterimi (sayfa numaraları ve benzerlik skorları)
- 🔒 Tamamen lokal çalışma (veri güvenliği)

## Gereksinimler

- Python 3.10+
- Ollama kurulu ve çalışır durumda
- Mistral modeli Ollama'da yüklü olmalı

## Kurulum

### 1. Ollama Kurulumu

Ollama'yı [ollama.ai](https://ollama.ai) adresinden indirip kurun.

Mistral modelini yükleyin:
```bash
ollama pull mistral
```

Ollama'nın çalıştığından emin olun:
```bash
ollama serve
```

### 2. Proje Kurulumu

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Environment dosyasını oluştur
cp .env.example .env

# Gerekirse .env dosyasını düzenleyin
```

### 3. Yapılandırma

`config.yaml` dosyasını ihtiyacınıza göre düzenleyebilirsiniz:
- Chunk boyutu
- Top-K değeri
- LLM modeli
- Embedding modeli

## Kullanım

### İnteraktif Mod

```bash
python app.py
```

Menüden seçim yapın:
1. PDF Yükle
2. Sohbet Başlat
3. Çıkış

### Komut Satırı Modu

```bash
# PDF yükle
python app.py load document.pdf

# PDF yükle ve sohbet başlat
python app.py load document.pdf --chat

# Önceden yüklenmiş PDF ile sohbet
python app.py chat
```

### Sohbet İçi Komutlar

- Normal soru sorun: `PDF'de ana konu nedir?`
- Çıkmak için: `/exit`, `/quit` veya `/çıkış`

## Proje Yapısı

```
rag-pdf-chatbot/
├── app.py                      # Ana CLI uygulaması
├── requirements.txt            # Python bağımlılıkları
├── config.yaml                 # Yapılandırma dosyası
├── .env                        # Environment variables
├── README.md                   # Bu dosya
├── PRD.md                      # Ürün gereksinimleri dokümanı
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF yükleme ve işleme
│   ├── text_splitter.py        # Chunking logic
│   ├── embeddings.py           # Embedding oluşturma
│   ├── vector_store.py         # ChromaDB işlemleri
│   ├── llm_handler.py          # Ollama LLM işlemleri
│   ├── rag_chain.py            # RAG pipeline
│   └── prompt_templates.py     # Prompt şablonları
│
├── data/
│   ├── uploads/                # Yüklenen PDF'ler (opsiyonel)
│   └── chroma_db/              # ChromaDB persist directory
│
└── tests/
    └── (test dosyaları)
```

## Teknik Detaylar

### Mimari

1. **PDF İşleme**: pdfplumber ile metin çıkarma
2. **Chunking**: LangChain RecursiveCharacterTextSplitter
3. **Embedding**: sentence-transformers (all-MiniLM-L6-v2)
4. **Vektör DB**: ChromaDB (cosine similarity)
5. **LLM**: Ollama (Mistral modeli)
6. **RAG Pipeline**: Query → Embed → Search → Generate

### Varsayılan Ayarlar

- Chunk size: 500 karakter
- Chunk overlap: 150 karakter
- Top-K: 5 chunk
- Temperature: 0.7
- Similarity threshold: 0.5

## Sorun Giderme

### Ollama Bağlantı Hatası

```
Ollama API'ye bağlanılamadı
```

**Çözüm**: Ollama'nın çalıştığından emin olun:
```bash
ollama serve
```

### Model Bulunamadı Hatası

```
Model 'mistral' bulunamadı
```

**Çözüm**: Modeli yükleyin:
```bash
ollama pull mistral
```

### PDF Metin Çıkarılamadı

Bazı PDF'ler görüntü tabanlıdır ve OCR gerektirebilir. Bu durumda:
- PDF'i OCR ile işleyin
- Veya metin tabanlı bir PDF kullanın

## Lisans

Bu proje eğitim amaçlıdır.

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen PR göndermeden önce kod standartlarına uyduğunuzdan emin olun.
