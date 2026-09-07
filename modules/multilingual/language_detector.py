import re
from typing import Dict, Any

class LanguageDetector:
    """Detects user input language with script heuristics and langdetect."""
    
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French"
    }
    
    LANGUAGE_FLAGS = {
        "en": "🇺🇸 English",
        "hi": "🇮🇳 Hindi (हिंदी)",
        "es": "🇪🇸 Spanish (Español)",
        "fr": "🇫🇷 French (Français)"
    }

    @classmethod
    def detect_language(cls, text: str) -> Dict[str, Any]:
        """Detects whether text is English, Hindi, Spanish, or French."""
        if not text or not text.strip():
            return {"code": "en", "name": "English", "confidence": 1.0, "is_supported": True}

        clean_text = text.strip()

        # Rule 1: Devanagari script detection for Hindi
        if re.search(r'[\u0900-\u097F]', clean_text):
            return {
                "code": "hi",
                "name": "Hindi",
                "display": cls.LANGUAGE_FLAGS["hi"],
                "confidence": 0.99,
                "is_supported": True
            }

        # Rule 2: Characteristic Spanish markers
        spanish_markers = ["¿", "¡", "ñ", "está", "pedido", "gracias", "hola", "por favor", "cancelar", "cuál", "cómo"]
        lower = clean_text.lower()
        if any(marker in lower for marker in spanish_markers):
            return {
                "code": "es",
                "name": "Spanish",
                "display": cls.LANGUAGE_FLAGS["es"],
                "confidence": 0.95,
                "is_supported": True
            }

        # Rule 3: Characteristic French markers
        french_markers = ["où", "commande", "merci", "bonjour", "s'il vous plaît", "livraison", "combien", "remboursement"]
        if any(marker in lower for marker in french_markers):
            return {
                "code": "fr",
                "name": "French",
                "display": cls.LANGUAGE_FLAGS["fr"],
                "confidence": 0.95,
                "is_supported": True
            }

        # Rule 4: Use langdetect library
        try:
            from langdetect import detect, detect_langs
            detected_langs = detect_langs(clean_text)
            if detected_langs:
                top = detected_langs[0]
                code = top.lang
                prob = top.prob
                if code in cls.SUPPORTED_LANGUAGES:
                    return {
                        "code": code,
                        "name": cls.SUPPORTED_LANGUAGES[code],
                        "display": cls.LANGUAGE_FLAGS[code],
                        "confidence": round(prob, 2),
                        "is_supported": True
                    }
        except Exception:
            pass

        # Default fallback is English
        return {
            "code": "en",
            "name": "English",
            "display": cls.LANGUAGE_FLAGS["en"],
            "confidence": 0.85,
            "is_supported": True
        }
