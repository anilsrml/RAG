"""
LLM Handler Modülü
Ollama ile lokal LLM işlemleri.
"""

import requests
from typing import Optional, Dict, Iterator, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLMHandler:
    """Ollama LLM yöneticisi"""
    
    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30
    ):
        """
        Args:
            model_name: Ollama model adı
            base_url: Ollama API base URL
            temperature: Model temperature
            max_tokens: Maksimum token sayısı
            timeout: Request timeout (saniye)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"
        
        # Modelin mevcut olup olmadığını kontrol et
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Modelin mevcut olup olmadığını kontrol eder"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if self.model_name not in model_names:
                    logger.warning(
                        f"Model '{self.model_name}' bulunamadı. "
                        f"Mevcut modeller: {', '.join(model_names)}"
                    )
                    logger.warning(f"Model yüklemek için: ollama pull {self.model_name}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API'ye bağlanılamadı: {e}")
            logger.error("Ollama'nın çalıştığından emin olun: ollama serve")
    
    def generate(
        self,
        prompt: str,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        LLM'den cevap üretir.
        
        Args:
            prompt: Gönderilecek prompt
            stream: Streaming response isteniyorsa True
            
        Returns:
            str veya Iterator[str]: Cevap metni veya streaming iterator
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                result = response.json()
                return result.get('response', '')
                
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM çağrısı başarısız: {e}")
            raise
    
    def _handle_streaming_response(self, response) -> Iterator[str]:
        """Streaming response'u işler"""
        for line in response.iter_lines():
            if line:
                try:
                    import json
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
    
    def test_connection(self) -> bool:
        """Ollama bağlantısını test eder"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

