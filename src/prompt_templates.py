"""
Prompt Şablonları Modülü
RAG için prompt şablonları.
"""

from typing import List, Dict


class PromptTemplates:
    """Prompt şablonları sınıfı"""
    
    @staticmethod
    def rag_prompt(context: str, question: str) -> str:
        """
        RAG için temel prompt şablonu.
        
        Args:
            context: Retrieved chunk'ların birleştirilmiş hali
            question: Kullanıcı sorusu
            
        Returns:
            str: Formatlanmış prompt
        """
        prompt = f"""Aşağıdaki dokümandan elde edilen bilgilere dayanarak soruyu cevapla.
Sadece verilen bilgileri kullan. Bilmiyorsan "Bu bilgi dokümanda yok" de.

Doküman İçeriği:
{context}

Soru: {question}

Cevap:"""
        return prompt
    
    @staticmethod
    def format_context(chunks: List[Dict], include_metadata: bool = True) -> str:
        """
        Retrieved chunk'ları context formatına dönüştürür.
        
        Args:
            chunks: Retrieved chunk'lar (metadata ile)
            include_metadata: Metadata'yı context'e dahil et
            
        Returns:
            str: Formatlanmış context metni
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, start=1):
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
    def format_sources(chunks: List[Dict]) -> List[Dict]:
        """
        Kaynak bilgilerini formatlar.
        
        Args:
            chunks: Retrieved chunk'lar
            
        Returns:
            List[Dict]: Formatlanmış kaynak listesi
        """
        sources = []
        
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

