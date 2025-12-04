# RAG PDF Chatbot (LangChain Entegrasyonu)

PDF dokümanlarını yükleyerek, lokalde çalışan bir dil modeli ile doküman içeriği hakkında soru sorabileceğiniz akıllı bir sohbet botu. LangChain framework'ü ile geliştirilmiştir.

## Özellikler

- 📄 PDF dosyalarını yükleme ve işleme (LangChain Document Loaders)
- 🔍 Semantik arama ile ilgili bilgileri bulma (LangChain Chroma)
- 💬 Lokal LLM (Ollama) ile sohbet (LangChain Ollama wrapper)
- 📚 Kaynak gösterimi (sayfa numaraları ve benzerlik skorları)
- 🔒 Tamamen lokal çalışma (veri güvenliği)
- 🧠 Sohbet geçmişi yönetimi (LangChain Memory)
- ⛓️ Modüler RAG chains (RetrievalQA ve ConversationalRetrievalChain)

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
- **Chunk boyutu**: 500 (varsayılan)
- **Top-K değeri**: 5 (varsayılan)
- **LLM modeli**: mistral (varsayılan)
- **Embedding modeli**: all-MiniLM-L6-v2 (varsayılan)
- **Chain type**: stuff, map_reduce, refine, map_rerank
- **Memory type**: buffer, window, summary
- **Memory enabled**: true/false

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
- Memory'yi temizlemek için: `/clear`

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
│   ├── pdf_processor.py        # PDF yükleme (LangChain PyPDFLoader)
│   ├── text_splitter.py        # Chunking logic (LangChain TextSplitter)
│   ├── embeddings.py           # Embedding (LangChain HuggingFaceEmbeddings)
│   ├── vector_store.py         # ChromaDB (LangChain Chroma wrapper)
│   ├── llm_handler.py          # Ollama LLM (LangChain Ollama wrapper)
│   ├── rag_chain.py            # RAG chains (RetrievalQA/ConversationalRetrievalChain)
│   ├── prompt_templates.py     # Prompt şablonları (LangChain PromptTemplate)
│   └── memory.py               # Conversation Memory (LangChain Memory)
│
├── data/
│   ├── uploads/                # Yüklenen PDF'ler (opsiyonel)
│   └── chroma_db/              # ChromaDB persist directory
│
└── tests/
    └── (test dosyaları)
```

## Teknik Detaylar

### Mimari (LangChain Framework)

1. **PDF İşleme**: LangChain PyPDFLoader
2. **Chunking**: LangChain RecursiveCharacterTextSplitter
3. **Embedding**: LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2)
4. **Vektör DB**: LangChain Chroma wrapper (cosine similarity)
5. **LLM**: LangChain Ollama wrapper (Mistral modeli)
6. **RAG Chains**: RetrievalQA (basit RAG) veya ConversationalRetrievalChain (memory ile)
7. **Memory**: ConversationBufferMemory, ConversationBufferWindowMemory veya ConversationSummaryMemory

### LangChain Entegrasyonu

Bu proje LangChain framework'ü kullanarak:
- **Standartlaşma**: LangChain'in standart API'lerini kullanır
- **Modülerlik**: Farklı LLM'ler ve vector store'lar kolayca değiştirilebilir
- **Memory Desteği**: Sohbet geçmişi otomatik yönetilir
- **Chain Flexibility**: Farklı RAG stratejileri (stuff, map_reduce, refine, map_rerank)
- **Production Ready**: LangChain'in production-ready özellikleri

### Varsayılan Ayarlar

- Chunk size: 500 karakter
- Chunk overlap: 150 karakter
- Top-K: 5 chunk
- Temperature: 0.7
- Chain type: stuff
- Memory type: buffer
- Memory enabled: true

## LangChain Chain Tipleri

### RetrievalQA (Basit RAG)
Tek soru-cevap için kullanılır. Memory devre dışı olduğunda aktiftir.
- **stuff**: Tüm dokümanları tek prompt'ta kullanır (hızlı, kısa dokümanlar için)
- **map_reduce**: Her dokümanı ayrı işler, sonra birleştirir (uzun dokümanlar için)
- **refine**: İteratif olarak cevabı iyileştirir
- **map_rerank**: Her doküman için skor verir, en iyisini seçer

### ConversationalRetrievalChain (Memory ile RAG)
Çoklu tur sohbet için kullanılır. Memory aktif olduğunda otomatik seçilir.
- Sohbet geçmişini tutar
- Bağlamsal sorular sorabilirsiniz
- "Bunu açıkla", "Daha fazla anlat" gibi takip soruları

## Memory Tipleri

- **buffer**: Tüm sohbet geçmişini tutar
- **window**: Son N mesajı tutar (performans için, config'de `window_size` ile ayarlanır)
- **summary**: Geçmişi özetler (uzun sohbetler için, LLM gerektirir)

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

### LangChain Import Hataları

Eğer import hataları alıyorsanız:
```bash
pip install --upgrade langchain langchain-community langchain-chroma
```

### Memory Çalışmıyor

Memory'yi devre dışı bırakmak için `config.yaml`:
```yaml
memory:
  enabled: false
```

## Lisans

Bu proje eğitim amaçlıdır.

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen PR göndermeden önce kod standartlarına uyduğunuzdan emin olun.
