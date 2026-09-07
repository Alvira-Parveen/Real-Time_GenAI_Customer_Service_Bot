import re
from typing import Dict, List, Set

class MedicalEntityExtractor:
    """Extracts clinical entities (symptoms, diseases, treatments, medications, body parts)."""

    # Curated medical entity vocabularies based on NIH UMLS semantic groups
    DISEASES_CONDITIONS = {
        "acromegaly", "gigantism", "diabetes", "type 2 diabetes", "type 1 diabetes",
        "addison's disease", "adrenal insufficiency", "cushing's syndrome",
        "hashimoto's disease", "hyperthyroidism", "hypothyroidism", "multiple endocrine neoplasia",
        "hyperparathyroidism", "cirrhosis", "celiac disease", "crohn's disease",
        "gallstones", "adhd", "arthritis", "osteoarthritis", "rheumatoid arthritis",
        "asthma", "autism", "cancer", "breast cancer", "colon cancer", "epilepsy",
        "hypertension", "high blood pressure", "angina", "arrhythmia", "atherosclerosis",
        "coronary heart disease", "aneurysm", "aarskog-scott syndrome", "aceruloplasminemia",
        "alpha-1 antitrypsin deficiency", "sleep apnea", "carpal tunnel syndrome"
    }

    SYMPTOMS = {
        "pain", "chest pain", "headache", "fatigue", "weakness", "swelling", "edema",
        "numbness", "shortness of breath", "dyspnea", "dizziness", "fever", "cough",
        "wheezing", "joint ache", "joint aches", "skin tags", "excessive sweating",
        "weight loss", "weight gain", "nausea", "vomiting", "blurred vision",
        "impaired vision", "skin rash", "coarse skin", "enlarged hands", "enlarged feet"
    }

    TREATMENTS_PROCEDURES = {
        "surgery", "radiation therapy", "chemotherapy", "dialysis", "biopsy",
        "screening", "blood test", "physical therapy", "diet modification",
        "lifestyle changes", "endoscopy", "colonoscopy", "mammogram", "mri", "ct scan",
        "ultrasound", "coronary bypass", "angioplasty", "stent"
    }

    MEDICATIONS = {
        "insulin", "metformin", "aspirin", "corticosteroids", "hydrocortisone",
        "beta blockers", "statins", "ace inhibitors", "antibiotics", "inhaler",
        "bronchodilators", "levothyroxine", "painkillers", "nsaids", "growth hormone antagonists"
    }

    BODY_PARTS = {
        "heart", "pituitary gland", "pituitary", "kidney", "kidneys", "liver",
        "lung", "lungs", "thyroid", "pancreas", "adrenal gland", "brain",
        "bone", "bones", "joints", "blood vessels", "arteries", "colon",
        "hands", "feet", "spine", "sinuses", "vocal cords"
    }

    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extracts recognized medical entities categorized by clinical type."""
        if not text:
            return {
                "diseases_and_conditions": [],
                "symptoms": [],
                "treatments": [],
                "medications": [],
                "body_parts": []
            }

        text_lower = text.lower()
        
        def match_vocab(vocab: Set[str]) -> List[str]:
            found = []
            for term in sorted(vocab, key=len, reverse=True):
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(term.title())
            return sorted(list(set(found)))

        return {
            "diseases_and_conditions": match_vocab(cls.DISEASES_CONDITIONS),
            "symptoms": match_vocab(cls.SYMPTOMS),
            "treatments": match_vocab(cls.TREATMENTS_PROCEDURES),
            "medications": match_vocab(cls.MEDICATIONS),
            "body_parts": match_vocab(cls.BODY_PARTS)
        }

    @classmethod
    def format_entities_for_display(cls, entities: Dict[str, List[str]]) -> str:
        """Formats detected entities into clean markdown bullets."""
        lines = []
        for cat, items in entities.items():
            if items:
                cat_name = cat.replace("_", " ").title()
                lines.append(f"- **{cat_name}**: {', '.join(items)}")
        if not lines:
            return "No specific medical entities extracted from input."
        return "\n".join(lines)
