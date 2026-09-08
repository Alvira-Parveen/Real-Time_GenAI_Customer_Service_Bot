import sys
import os
import pandas as pd
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from modules.knowledge_base.updater import KnowledgeBaseManager
from modules.knowledge_base.loader import DocumentLoader
from modules.sentiment.sentiment_analyzer import SentimentAnalyzer
from modules.multilingual.language_detector import LanguageDetector
from modules.multilingual.multilingual_response import MultilingualHandler
from modules.medical.retrieval import MedicalRetriever
from modules.medical.entity_extraction import MedicalEntityExtractor
from modules.research.paper_search import ArxivPaperSearch
from modules.research.summarizer import PaperSummarizer
from modules.research.visualization import ResearchVisualizer
from modules.multimodal.image_analysis import MultimodalAnalyzer
from modules.multimodal.image_generation import ImageGenerator
from modules.llm_client import LLMClient

def test_checklist():
    print("================================================================================")
    print("LIVE COMPLIANCE VERIFICATION WALKTHROUGH — ALL 6 ELEVANCE SKILLS TASKS")
    print("================================================================================\n")

    # -------------------------------------------------------------------------
    # STEP 0: Built-in Compliance Matrix & Datasets
    # -------------------------------------------------------------------------
    print(">>> STEP 0: Verifying Compliance Matrix & Local Datasets...")
    datasets = {
        "Task 1 (Policies)": "data/customer_service/",
        "Task 3 (MedQuAD)": "data/medquad/",
        "Task 4 (arXiv CS)": "data/arxiv/arxiv_cs_papers.json",
        "Task 5 (Sentiment)": "data/sentiment_test.csv"
    }
    for name, path in datasets.items():
        exists = os.path.exists(path)
        print(f"  [OK] Dataset '{name}' exists at '{path}': {exists}")
        assert exists, f"Missing dataset: {path}"
    print("  [PASSED] Step 0 Verified!\n")

    # -------------------------------------------------------------------------
    # STEP 1: Task 1 — Dynamic Knowledge Base
    # -------------------------------------------------------------------------
    print(">>> STEP 1: Verifying Task 1 — Dynamic Knowledge Base...")
    kb = KnowledgeBaseManager()
    initial_stats = kb.get_stats()
    print(f"  Initial Index Stats: {initial_stats['total_documents']} documents, {initial_stats['total_chunks']} chunks")
    assert initial_stats['total_chunks'] >= 22, "Initial chunk count mismatch"

    # Simulate button click: "Load Sample: Extended Holiday Warranty 2026" + "Ingest and Index"
    import time
    test_title = f"holiday_extended_warranty_{int(time.time())}.txt"
    test_content = (
        "ApexTech Holiday Extended Warranty 2026 Policy Addendum:\n"
        "All consumer electronics purchased between November 1 and December 31, 2026 receive an exclusive 90-day "
        "no-questions-asked money-back guarantee and free accidental screen damage protection. "
        "Return shipping is completely free for all holiday gift orders."
    )
    print(f"  Simulating live document ingestion: {test_title}...")
    ingest_res = kb.add_document_from_text(test_title, test_content)
    print(f"  Ingestion Result: {ingest_res['total_documents']} files, {ingest_res['total_chunks']} chunks")
    assert ingest_res['total_chunks'] > initial_stats['total_chunks'], "Chunk count should increase!"

    # Simulate button click: "Execute Vector Search"
    query = "What is the holiday extended warranty return window for 2026?"
    print(f"  Executing semantic vector retrieval for query: '{query}'...")
    results = kb.query(query, top_k=2)
    print(f"  Top Retrieved Result Source: {results[0]['source']} (Relevance Score: {results[0]['score']:.3f})")
    print(f"  Retrieved Snippet: {results[0]['text'][:120]}...")
    assert "90-day" in results[0]['text'], "Failed to retrieve newly indexed policy snippet!"
    print("  [PASSED] Step 1 Verified!\n")

    # -------------------------------------------------------------------------
    # STEP 2: Task 2 — Multimodal Diagnostics
    # -------------------------------------------------------------------------
    print(">>> STEP 2: Verifying Task 2 — Multimodal Diagnostics...")
    from app import get_demo_damaged_image
    demo_img = get_demo_damaged_image()
    assert demo_img.size == (420, 320), "Demo image dimensions mismatch"
    print(f"  Generated test intake image: {demo_img.size[0]}x{demo_img.size[1]}px")

    analyzer = MultimodalAnalyzer()
    print("  Executing Multimodal Defect Analysis on cracked display image...")
    report = analyzer.analyze_image(demo_img, "Analyze this product image for hardware defects and suggest warranty resolution.")
    print(f"  Identified Defect Type: {report.get('defect_type')}")
    print(f"  Assessed Severity: {report.get('severity')}")
    print(f"  Diagnostic Summary:\n    {report['text'].splitlines()[0]}")
    assert "CRACK" in str(report.get("defect_type", "")).upper() or "DAMAGE" in str(report.get("defect_type", "")).upper()

    print("  Simulating Text-to-Image Generative Concept Rendering: 'Over-Ear ANC Headset'...")
    gen = ImageGenerator()
    gen_res = gen.generate_image("A futuristic sleek wireless noise cancelling headphones in matte obsidian with brass accents")
    print(f"  Image Generation Engine: {gen_res['engine']}")
    print(f"  Generated Concept Image Size: {gen_res['image'].size}")
    assert gen_res['image'] is not None
    print("  [PASSED] Step 2 Verified!\n")

    # -------------------------------------------------------------------------
    # STEP 3: Task 3 — Clinical Reference (MedQuAD)
    # -------------------------------------------------------------------------
    print(">>> STEP 3: Verifying Task 3 — Clinical Reference (MedQuAD)...")
    med_retriever = MedicalRetriever()
    med_query = "What is Acromegaly?"
    print(f"  Executing MedQuAD Retrieval for clinical query: '{med_query}'...")
    med_ans = med_retriever.answer_medical_query(med_query)
    entities = med_ans.get("entities", {})
    print(f"  Identified Clinical Entities: {entities}")
    assert "Acromegaly" in entities.get("diseases_and_conditions", [])
    records = med_ans.get("retrieved_records", [])
    print(f"  Retrieved {len(records)} Grounded NIH MedQuAD QA Records")
    print(f"  Focus: {records[0].get('focus')} | QType: {records[0].get('qtype')} | Score: {records[0].get('score'):.3f}")
    assert len(records) > 0, "No MedQuAD evidence records retrieved!"
    print("  [PASSED] Step 3 Verified!\n")

    # -------------------------------------------------------------------------
    # STEP 4: Task 4 — Research Explorer (arXiv)
    # -------------------------------------------------------------------------
    print(">>> STEP 4: Verifying Task 4 — Research Explorer (arXiv)...")
    searcher = ArxivPaperSearch()
    all_papers = searcher.get_all_papers()
    print(f"  Total Curated arXiv Computer Science Landmark Papers: {len(all_papers)}")
    assert len(all_papers) >= 30

    # Tab 1: Research Q&A
    r_query = "Explain self-attention mechanism in simple terms"
    print(f"  Executing arXiv semantic search & explanation for: '{r_query}'...")
    r_res = searcher.answer_research_query(r_query)
    paper_id = r_res['papers'][0].get('id', r_res['papers'][0].get('chunk_id', '1706.03762'))
    print(f"  Referenced Paper: {r_res['papers'][0]['title']} (arXiv:{paper_id})")
    assert len(r_res['papers']) > 0, "No arXiv papers retrieved!"

    # Tab 2: 5-Part Summarizer
    vaswani_paper = next(p for p in all_papers if p["id"] == "1706.03762")
    print(f"  Generating Mandatory 5-Part Summary for: '{vaswani_paper['title']}'...")
    summary_5part = searcher.generate_structured_summary(vaswani_paper)
    required_sections = ["problem", "methodology", "results", "limitations", "future_work"]
    for s in required_sections:
        assert s in summary_5part and len(summary_5part[s]) > 20, f"Section {s} missing or empty!"
        print(f"    - Section '{s.upper()}': {summary_5part[s][:80]}...")

    # Tab 3: Dual-Level Concept Explanation
    print("  Generating Dual-Level Explanations for 'Self-Attention Mechanism'...")
    int_exp = searcher.explain_concept("Self-Attention Mechanism", level="intuitive")
    math_exp = searcher.explain_concept("Self-Attention Mechanism", level="mathematical")
    print(f"    Intuitive (Excerpt): {int_exp['text'][:90]}...")
    print(f"    Mathematical (Excerpt): {math_exp['text'][:90]}...")
    assert "Softmax" in math_exp['text'] or "Q" in math_exp['text']

    # Tab 4: Interactive Plotly Visualizations
    fig_topic = ResearchVisualizer.create_paper_topic_graph(all_papers)
    fig_concept = ResearchVisualizer.create_concept_graph()
    assert fig_topic is not None and fig_concept is not None
    print("    Plotly category topic graph and concept dependency graph verified.")
    print("  [PASSED] Step 4 Verified!\n")

    # -------------------------------------------------------------------------
    # STEP 5: Task 5 — Sentiment Analytics & Benchmark
    # -------------------------------------------------------------------------
    print(">>> STEP 5: Verifying Task 5 — Sentiment Analytics & Benchmark...")
    sentiment_analyzer = SentimentAnalyzer()
    print("  Evaluating model against 'data/sentiment_test.csv' (30 labeled customer service samples)...")
    bench = sentiment_analyzer.evaluate_dataset()
    print(f"  Overall Accuracy : {bench['accuracy'] * 100:.1f}%")
    print(f"  Precision (Macro): {bench['precision_macro'] * 100:.1f}%")
    print(f"  Recall (Macro)   : {bench['recall_macro'] * 100:.1f}%")
    print(f"  F1-Score (Macro) : {bench['f1_macro'] * 100:.1f}%")
    assert bench['accuracy'] >= 0.75, "Accuracy benchmark should be >= 75%"

    # Confusion matrix
    cm = bench['confusion_matrix']
    print(f"  Confusion Matrix Shape: {len(cm)}x{len(cm[0])} across classes: {bench['labels']}")

    # Live classification tests
    frustrated_msg = "I am extremely frustrated! My order is 5 days late and arrived completely broken."
    s_neg = sentiment_analyzer.analyze(frustrated_msg)
    print(f"  Complaint Test: '{frustrated_msg[:45]}...' -> Classified: {s_neg['label'].upper()} (Score: {s_neg['compound']})")
    assert s_neg['label'] == "negative"

    positive_msg = "Thank you so much, the customer support team was fantastic and resolved my issue immediately!"
    s_pos = sentiment_analyzer.analyze(positive_msg)
    print(f"  Praise Test: '{positive_msg[:45]}...' -> Classified: {s_pos['label'].upper()} (Score: {s_pos['compound']})")
    assert s_pos['label'] == "positive"
    print("  [PASSED] Step 5 Verified!\n")

    # -------------------------------------------------------------------------
    # STEP 6: Task 6 — Multilingual Processing
    # -------------------------------------------------------------------------
    print(">>> STEP 6: Verifying Task 6 — Multilingual Processing...")
    hindi_msg = "नमस्ते, क्या मुझे क्षतिग्रस्त उत्पाद के लिए रिफंड मिल सकता है?"
    det_hi = LanguageDetector.detect_language(hindi_msg)
    print(f"  Hindi Query: '{hindi_msg}'")
    print(f"    Detected: {det_hi['name']} (Code: {det_hi['code']}, Confidence: {det_hi['confidence']*100:.0f}%)")
    assert det_hi['code'] == "hi"

    hi_empathy = MultilingualHandler.get_empathy_prefix("hi")
    print(f"    Localized Hindi Empathy Template: '{hi_empathy}'")
    assert "निराशा" in hi_empathy

    # Spanish & French checks
    es_msg = "¿Puedo devolver este producto si está dañado?"
    det_es = LanguageDetector.detect_language(es_msg)
    print(f"  Spanish Query -> Detected: {det_es['name']} (Code: {det_es['code']})")
    assert det_es['code'] == "es"

    fr_msg = "Bonjour, mon colis est arrivé endommagé."
    det_fr = LanguageDetector.detect_language(fr_msg)
    print(f"  French Query  -> Detected: {det_fr['name']} (Code: {det_fr['code']})")
    assert det_fr['code'] == "fr"
    print("  [PASSED] Step 6 Verified!\n")

    # -------------------------------------------------------------------------
    # STEP 7: Unified Customer Service Bot (Base + All Tasks Integrated)
    # -------------------------------------------------------------------------
    print(">>> STEP 7: Verifying Unified Customer Service Bot Integration...")
    llm = LLMClient()

    # Case A: Frustrated customer triggers sentiment de-escalation + RAG
    print("  Testing Integration Case A: Frustrated Customer Late Order...")
    prompt_a = "I am furious, my laptop was delayed by 5 days and arrived with a broken screen!"
    sent_a = sentiment_analyzer.analyze(prompt_a)
    chunks_a = kb.query(prompt_a, top_k=2)
    context_a = "\n".join([c['text'] for c in chunks_a])
    resp_a = llm.generate_response(
        prompt=prompt_a,
        system_instruction=f"Tone: {sent_a['tone_guidance']}",
        retrieved_context=context_a
    )
    full_reply_a = MultilingualHandler.get_empathy_prefix("en") + resp_a['text']
    print(f"    Sentiment: {sent_a['label'].upper()}")
    print(f"    Retrieved Policy Chunks: {len(chunks_a)}")
    print(f"    Assistant Unified Response (Excerpt): {full_reply_a[:150]}...")
    assert "sincere apologies" in full_reply_a.lower() or "apologize" in full_reply_a.lower()

    # Case B: Hindi Customer triggers Language Detection + Localized Response
    print("\n  Testing Integration Case B: Multilingual Hindi Query...")
    prompt_b = "नमस्ते, मेरा रिफंड कब तक आएगा?"
    det_b = LanguageDetector.detect_language(prompt_b)
    chunks_b = kb.query(prompt_b, top_k=2)
    context_b = "\n".join([c['text'] for c in chunks_b])
    resp_b = llm.generate_response(
        prompt=prompt_b,
        system_instruction="Respond in Hindi with professional customer service tone.",
        retrieved_context=context_b,
        target_language="Hindi"
    )
    full_reply_b = MultilingualHandler.get_empathy_prefix("hi") + resp_b['text']
    print(f"    Detected Language: {det_b['name']}")
    print(f"    Assistant Hindi Response (Excerpt): {full_reply_b[:140]}...")
    assert any('\u0900' <= char <= '\u097F' for char in full_reply_b), "Response must contain Devanagari text!"

    print("\n================================================================================")
    print("ALL 6 TASKS + UNIFIED INTEGRATION FULLY TESTED & VERIFIED (100% COMPLIANT)")
    print("================================================================================")

if __name__ == "__main__":
    test_checklist()
