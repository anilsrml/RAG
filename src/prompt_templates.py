"""
Prompt Şablonları Modülü
RAG için prompt şablonları.
LangChain PromptTemplate kullanır.
"""

from typing import List, Dict
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document


class PromptTemplates:
    """Prompt şablonları sınıfı - LangChain PromptTemplate kullanır"""
    
    def __init__(self):
        """LangChain prompt template'lerini oluştur"""
        # RAG için PromptTemplate
        self.rag_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Aşağıdaki dokümandan elde edilen bilgilere dayanarak soruyu cevapla.
Sadece verilen bilgileri kullan. Bilmiyorsan "Bu bilgi dokümanda yok" de.

Doküman İçeriği:
{context}

Soru: {question}

Cevap:"""
        )
        
        # Chat için ChatPromptTemplate
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", "Aşağıdaki dokümandan elde edilen bilgilere dayanarak soruları cevapla. Sadece verilen bilgileri kullan."),
            ("human", "Doküman İçeriği:\n{context}\n\nSoru: {question}")
        ])
    
    def rag_prompt(self, context: str, question: str) -> str:
        """
        RAG için temel prompt şablonu.
        
        Args:
            context: Retrieved chunk'ların birleştirilmiş hali
            question: Kullanıcı sorusu
            
        Returns:
            str: Formatlanmış prompt
        """
        return self.rag_template.format(context=context, question=question)
    
    def get_rag_template(self) -> PromptTemplate:
        """
        LangChain PromptTemplate objesini döndürür (Chains için).
        
        Returns:
            PromptTemplate: LangChain PromptTemplate objesi
        """
        return self.rag_template
    
    def get_chat_template(self) -> ChatPromptTemplate:
        """
        LangChain ChatPromptTemplate objesini döndürür (Chat chains için).
        
        Returns:
            ChatPromptTemplate: LangChain ChatPromptTemplate objesi
        """
        return self.chat_template
    
    @staticmethod
    def format_context(chunks: List[Dict], include_metadata: bool = True) -> str:
        """
        Retrieved chunk'ları context formatına dönüştürür.
        
        Args:
            chunks: Retrieved chunk'lar (metadata ile) veya LangChain Document listesi
            include_metadata: Metadata'yı context'e dahil et
            
        Returns:
            str: Formatlanmış context metni
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, start=1):
            # LangChain Document objesi mi kontrol et
            if isinstance(chunk, Document):
                text = chunk.page_content
                metadata = chunk.metadata
            else:
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
            
            if include_metadata:
                source = metadata.get('source_file', 'Bilinmeyen')
                page = metadata.get('page', '?')
                context_parts.append(f"[Kaynak {i} - {source}, Sayfa {page}]\n{text}")
            else:
                context_parts.append(text)
        
        return "\n\n---\n\n".join(context_parts)
    
    @staticmethod
    def format_sources(chunks: List[Dict], documents: List[Document] = None) -> List[Dict]:
        """
        Kaynak bilgilerini formatlar.
        
        Args:
            chunks: Retrieved chunk'lar (Dict formatında)
            documents: LangChain Document listesi (opsiyonel)
            
        Returns:
            List[Dict]: Formatlanmış kaynak listesi
        """
        sources = []
        
        # Eğer documents verilmişse onu kullan
        if documents:
            for doc in documents:
                metadata = doc.metadata
                sources.append({
                    'source_file': metadata.get('source_file', 'Bilinmeyen'),
                    'page': metadata.get('page', '?'),
                    'text_snippet': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'similarity': None,
                    'chunk_id': metadata.get('chunk_id')
                })
            return sources
        
        # Eski format için
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            distance = chunk.get('distance')
            
            # Cosine distance'ı similarity'ye çevir (1 - distance)
            similarity = 1 - distance if distance is not None else None
            
            sources.append({
                'source_file': metadata.get('source_file', 'Bilinmeyen'),
                'page': metadata.get('page', '?'),
                'text_snippet': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                'similarity': round(similarity, 3) if similarity is not None else None,
                'chunk_id': metadata.get('chunk_id')
            })
        
        return sources

