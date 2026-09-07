from typing import Dict, Any, List

class MultilingualHandler:
    """Handles multilingual response synthesis, query translation, and localized templates."""

    GREETINGS = {
        "en": "Hello! How can I assist you with ApexTech services today?",
        "hi": "नमस्ते! मैं आज आपकी अपेक्सटेक सेवाओं में कैसे सहायता कर सकता हूँ?",
        "es": "¡Hola! ¿Cómo puedo ayudarle hoy con los servicios de ApexTech?",
        "fr": "Bonjour ! Comment puis-je vous aider aujourd'hui avec les services ApexTech ?"
    }

    EMPATHY_PREFIX = {
        "en": "I completely understand your frustration, and I apologize for any inconvenience caused. Let me help you resolve this right away:\n\n",
        "hi": "मैं आपकी निराशा को पूरी तरह समझता हूँ और हुई किसी भी असुविधा के लिए क्षमा चाहता हूँ। मुझे इसे तुरंत हल करने में आपकी सहायता करने दें:\n\n",
        "es": "Entiendo perfectamente su frustración y le pido sinceras disculpas por las molestias causadas. Permítame ayudarle a resolver esto de inmediato:\n\n",
        "fr": "Je comprends tout à fait votre frustration et vous présente mes excuses pour ce désagrément. Laissez-moi vous aider à résoudre ce problème immédiatement :\n\n"
    }

    ORDER_TRACKING_TEMPLATES = {
        "en": "To track your order, please provide your order ID (e.g., #ORD-1234). Orders placed with standard shipping typically arrive in 3-5 business days.",
        "hi": "अपने ऑर्डर को ट्रैक करने के लिए कृपया अपनी ऑर्डर आईडी (जैसे #ORD-1234) प्रदान करें। मानक शिपिंग 3-5 कार्य दिवसों में पहुंचती है।",
        "es": "Para rastrear su pedido, proporcione su número de pedido (ej. #ORD-1234). El envío estándar suele tardar de 3 a 5 días hábiles.",
        "fr": "Pour suivre votre commande, veuillez indiquer votre numéro de commande (ex. #ORD-1234). La livraison standard prend 3 à 5 jours ouvrables."
    }

    DISCLAIMER_MEDICAL = {
        "en": "⚠️ Educational Disclaimer: This information is based on the MedQuAD dataset and is not a substitute for professional medical advice, diagnosis, or emergency care.",
        "hi": "⚠️ शैक्षिक अस्वीकरण: यह जानकारी MedQuAD डेटासेट पर आधारित है और पेशेवर चिकित्सा सलाह, निदान या आपातकालीन देखभाल का विकल्प नहीं है।",
        "es": "⚠️ Descargo de responsabilidad educativa: Esta información se basa en el conjunto de datos MedQuAD y no sustituye el consejo, diagnóstico o atención médica profesional.",
        "fr": "⚠️ Avertissement éducatif : Ces informations sont basées sur le jeu de données MedQuAD et ne remplacent pas un avis médical professionnel, un diagnostic ou des soins d'urgence."
    }

    @classmethod
    def get_greeting(cls, lang_code: str) -> str:
        return cls.GREETINGS.get(lang_code, cls.GREETINGS["en"])

    @classmethod
    def get_empathy_prefix(cls, lang_code: str) -> str:
        return cls.EMPATHY_PREFIX.get(lang_code, cls.EMPATHY_PREFIX["en"])

    @classmethod
    def get_medical_disclaimer(cls, lang_code: str) -> str:
        return cls.DISCLAIMER_MEDICAL.get(lang_code, cls.DISCLAIMER_MEDICAL["en"])
