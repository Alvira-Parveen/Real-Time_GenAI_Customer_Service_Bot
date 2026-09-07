import os
from typing import List, Dict, Any, Optional

class LLMClient:
    """Unified LLM client for Google Gemini models with robust local synthesis fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = None
        self.model_name = "gemini-2.0-flash"
        self._init_client()

    def set_api_key(self, api_key: str):
        """Allows dynamic API key configuration from Streamlit sidebar."""
        self.api_key = api_key
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            self.client = None
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.sdk = "genai"
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self.client = gai.GenerativeModel("gemini-1.5-flash")
                self.sdk = "generativeai"
            except Exception as e:
                print(f"Gemini initialization error: {e}")
                self.client = None

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = "",
        conversation_history: List[Dict[str, str]] = None,
        retrieved_context: str = "",
        target_language: str = "English"
    ) -> Dict[str, Any]:
        """Generates a contextual response using Gemini or local knowledge synthesizer."""
        full_system = system_instruction
        if target_language and target_language != "English":
            full_system += f"\nRespond fluently in {target_language}."

        if retrieved_context:
            full_system += f"\nUse the following verified reference knowledge:\n{retrieved_context}\nGround your response strictly in the provided references."

        # Format conversation history
        history_text = ""
        if conversation_history:
            turns = []
            for msg in conversation_history[-4:]: # Last 4 turns
                role = "User" if msg["role"] == "user" else "Assistant"
                turns.append(f"{role}: {msg['content']}")
            history_text = "\n".join(turns) + "\n"

        full_prompt = f"{history_text}User: {prompt}\nAssistant:"

        # Try Gemini API if client is available
        if self.client and self.api_key:
            try:
                if self.sdk == "genai":
                    combined = f"{full_system}\n\n{full_prompt}"
                    resp = self.client.models.generate_content(
                        model=self.model_name,
                        contents=combined
                    )
                    return {
                        "text": resp.text,
                        "model": self.model_name,
                        "engine": "Google Gemini 2.0 Flash"
                    }
                else:
                    resp = self.client.generate_content(f"{full_system}\n\n{full_prompt}")
                    return {
                        "text": resp.text,
                        "model": "gemini-1.5-flash",
                        "engine": "Google Gemini 1.5 Flash"
                    }
            except Exception as e:
                print(f"Gemini API request notice ({e}). Switching to local synthesizer.")

        # Local Synthesis Engine (deterministic, grounded, zero external dependencies)
        local_reply = self._local_grounded_synthesis(
            prompt=prompt,
            retrieved_context=retrieved_context,
            target_language=target_language,
            system_instruction=system_instruction
        )
        return {
            "text": local_reply,
            "model": "Local-Grounded-Synthesizer",
            "engine": "Local RAG Synthesis Engine (Gemini API key optional)"
        }

    def _local_grounded_synthesis(
        self,
        prompt: str,
        retrieved_context: str,
        target_language: str,
        system_instruction: str
    ) -> str:
        """Synthesizes high-quality, professional grounded answers without an active API key."""
        lang_lower = target_language.lower() if target_language else "english"

        # Check if medical mode
        if "MedQuAD" in system_instruction or "medical" in system_instruction.lower():
            if not retrieved_context:
                msg = "I do not have specific verified information in the MedQuAD dataset regarding this inquiry. Please consult a licensed medical specialist."
            else:
                msg = f"Based on verified medical records from the MedQuAD dataset:\n\n{retrieved_context}\n\n*Please discuss these findings with a qualified healthcare professional.*"
            return msg

        # Check if research mode
        if "arXiv" in system_instruction or "research" in system_instruction.lower():
            if not retrieved_context:
                return "No matching Computer Science papers were found for this query in the local arXiv index. Try querying topics like 'Transformer architectures', 'Self-Attention', 'Vision Transformers', or 'LoRA'."
            return f"Based on indexed Computer Science research from arXiv:\n\n{retrieved_context}\n\n*Feel free to ask follow-up questions regarding method details or mathematical explanations!*"

        # Customer Service Mode
        if retrieved_context:
            base_reply = f"Here is the official information regarding your request:\n\n{retrieved_context}\n\nIs there anything else I can help you with regarding your order or account?"
        else:
            base_reply = "Thank you for reaching out to ApexTech Customer Service. I am here to help with your orders, returns, warranty, and technical product support. Could you please specify your order number or product model?"

        # Local language adaptation for common languages
        if "hindi" in lang_lower:
            return f"नमस्ते! अपेक्सटेक ग्राहक सेवा से जानकारी:\n\n{retrieved_context if retrieved_context else 'हम आपके ऑर्डर, वारंटी और रिटर्न में सहायता के लिए उपलब्ध हैं। कृपया अपना ऑर्डर नंबर बताएं।'}\n\nक्या मैं आपकी किसी और चीज़ में मदद कर सकता हूँ?"
        elif "spanish" in lang_lower:
            return f"¡Hola! Información del servicio al cliente de ApexTech:\n\n{retrieved_context if retrieved_context else 'Estamos a su disposición para ayudarle con sus pedidos, devoluciones y garantía.'}\n\n¿Hay algo más en lo que pueda ayudarle hoy?"
        elif "french" in lang_lower:
            return f"Bonjour ! Informations du service client ApexTech :\n\n{retrieved_context if retrieved_context else 'Nous sommes à votre disposition pour vous aider avec vos commandes, retours et garantie.'}\n\nComment puis-je vous aider davantage aujourd'hui ?"

        return base_reply
