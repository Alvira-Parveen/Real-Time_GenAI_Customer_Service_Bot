import os
import unittest
from PIL import Image

# Import tasks
from modules.knowledge_base.updater import KnowledgeBaseManager
from modules.knowledge_base.loader import DocumentLoader
from modules.sentiment.sentiment_analyzer import SentimentAnalyzer
from modules.multilingual.language_detector import LanguageDetector
from modules.multilingual.multilingual_response import MultilingualHandler
from modules.medical.entity_extraction import MedicalEntityExtractor
from modules.medical.retrieval import MedicalRetriever
from modules.research.paper_search import ArxivPaperSearch
from modules.research.summarizer import PaperSummarizer
from modules.research.visualization import ResearchVisualizer
from modules.multimodal.image_analysis import MultimodalAnalyzer
from modules.multimodal.image_generation import ImageGenerator

class TestInternshipBotTasks(unittest.TestCase):

    def test_task1_knowledge_base(self):
        print("\n--- Testing Task 1: Dynamic Knowledge Base ---")
        kb = KnowledgeBaseManager()
        stats = kb.get_stats()
        self.assertGreater(stats["total_documents"], 0)
        self.assertGreater(stats["total_chunks"], 0)
        
        # Test query
        results = kb.query("What is the return window?", top_k=2)
        self.assertGreater(len(results), 0)
        print(f"Task 1 Success: {stats['total_chunks']} chunks indexed across {stats['total_documents']} files.")

    def test_task2_multimodal(self):
        print("\n--- Testing Task 2: Multimodal Chatbot ---")
        analyzer = MultimodalAnalyzer()
        test_img = Image.new("RGB", (100, 100), color=(150, 80, 50))
        report = analyzer.analyze_image(test_img, "Identify defects")
        self.assertTrue(report["success"])
        self.assertIn("Visual Diagnostic Assessment", report["text"])

        # Test image generation fallback/procedural
        gen = ImageGenerator()
        gen_res = gen.generate_image("Modern ANC headphones")
        self.assertTrue(gen_res["success"])
        self.assertIsNotNone(gen_res["image"])
        print("Task 2 Success: Multimodal vision diagnostic and image generator functional.")

    def test_task3_medical_qa(self):
        print("\n--- Testing Task 3: Medical Q&A with MedQuAD ---")
        # Entity extraction
        text = "Patient complains of fatigue, swelling, and severe joint aches caused by acromegaly."
        ents = MedicalEntityExtractor.extract_entities(text)
        self.assertIn("Acromegaly", ents["diseases_and_conditions"])
        self.assertTrue(len(ents["symptoms"]) > 0)

        # Retrieval
        retriever = MedicalRetriever()
        res = retriever.retrieve("What are the symptoms of acromegaly?", top_k=1)
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0]["focus"], "Acromegaly")
        print(f"Task 3 Success: Extracted {ents}, retrieved MedQuAD QA with score {res[0]['score']:.2f}")

    def test_task4_scientific_research(self):
        print("\n--- Testing Task 4: Scientific Expert Chatbot ---")
        searcher = ArxivPaperSearch()
        papers = searcher.search_papers("transformer architectures for vision", top_k=2)
        self.assertGreater(len(papers), 0)
        
        top_paper = papers[0]
        # Structured summary test
        summary = PaperSummarizer.structure_summary_from_abstract(
            top_paper["title"],
            top_paper["abstract"],
            top_paper["authors"]
        )
        self.assertIn("problem", summary)
        self.assertIn("approach", summary)
        self.assertIn("main_idea", summary)
        self.assertIn("results", summary)
        self.assertIn("conclusion", summary)

        # Visualization test
        fig = ResearchVisualizer.create_paper_topic_graph(searcher.get_all_papers())
        self.assertIsNotNone(fig)
        print("Task 4 Success: arXiv paper search, structured summarizer, and Plotly graph verified.")

    def test_task5_sentiment_analysis(self):
        print("\n--- Testing Task 5: Sentiment Analysis & Evaluation ---")
        analyzer = SentimentAnalyzer()
        
        # Test individual sentiment classification
        pos = analyzer.analyze("I absolutely love this product, fast shipping and perfect quality!")
        self.assertEqual(pos["label"], "positive")

        neg = analyzer.analyze("My order arrived broken and customer support was completely unhelpful.")
        self.assertEqual(neg["label"], "negative")

        # Test quantitative evaluation on CSV
        metrics = analyzer.evaluate_dataset("data/sentiment_test.csv")
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreater(metrics["accuracy"], 0.70)
        print(f"Task 5 Success: Sentiment accuracy = {metrics['accuracy']*100:.1f}%, F1 = {metrics['f1_macro']*100:.1f}%")

    def test_task6_multilingual(self):
        print("\n--- Testing Task 6: Multilingual Chatbot ---")
        # Language detection
        en_res = LanguageDetector.detect_language("Can I exchange this item within 30 days?")
        self.assertEqual(en_res["code"], "en")

        es_res = LanguageDetector.detect_language("¿Puedo devolver este producto si está dañado?")
        self.assertEqual(es_res["code"], "es")

        hi_res = LanguageDetector.detect_language("नमस्ते, क्या मुझे रिफंड मिल सकता है?")
        self.assertEqual(hi_res["code"], "hi")

        # Empathy prefix
        hi_prefix = MultilingualHandler.get_empathy_prefix("hi")
        self.assertIn("असुविधा", hi_prefix)
        print("Task 6 Success: Multilingual detection and localized empathy verified.")

if __name__ == "__main__":
    unittest.main()
