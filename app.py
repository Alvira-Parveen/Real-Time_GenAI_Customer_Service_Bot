import os
import streamlit as st
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project modules
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

# Streamlit Page Config
st.set_page_config(
    page_title="GenAI Customer Service Bot — Extended",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Modern Aesthetics
st.markdown("""
<style>
    /* Main Layout Styling */
    .main {
        background-color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 6px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-positive { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-negative { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .badge-neutral { background-color: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
    .badge-lang { background-color: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe; }
    
    /* Disclaimer Card */
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 18px;
        border-radius: 6px;
        color: #92400e;
        font-size: 13px;
        margin-bottom: 18px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# App State Caching
@st.cache_resource
def get_kb_manager():
    return KnowledgeBaseManager()

@st.cache_resource
def get_sentiment_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def get_medical_retriever():
    return MedicalRetriever()

@st.cache_resource
def get_arxiv_searcher():
    return ArxivPaperSearch()

# Initialize Singletons
kb_manager = get_kb_manager()
sentiment_analyzer = get_sentiment_analyzer()
medical_retriever = get_medical_retriever()
arxiv_searcher = get_arxiv_searcher()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "medical_messages" not in st.session_state:
    st.session_state.medical_messages = []
if "research_messages" not in st.session_state:
    st.session_state.research_messages = []

# ==============================================================================
# SIDEBAR NAVIGATION & SETTINGS
# ==============================================================================
with st.sidebar:
    st.markdown("## 🤖 GENAI BOT EXTENDED")
    st.caption("Real-Time GenAI Customer Service Bot with 6 Internship Extensions")
    st.divider()

    # Mode Selector
    mode = st.radio(
        "🎯 Select Operational Mode:",
        ["Customer Service", "Medical Q&A (MedQuAD)", "Research Expert (arXiv)"],
        index=0
    )

    st.divider()
    
    # Language Selector (Task 6)
    st.markdown("### 🌐 Language Settings")
    lang_choice = st.selectbox(
        "Interaction Language:",
        ["Auto Detect", "English", "Hindi", "Spanish", "French"],
        index=0
    )
    manual_lang_code = {
        "Auto Detect": None,
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr"
    }[lang_choice]

    st.divider()

    # API Configuration
    st.markdown("### 🔑 API Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "")
    user_api_key = st.text_input(
        "Google Gemini API Key:",
        value=env_key,
        type="password",
        help="Used for Google Gemini multimodal vision, Imagen-3, and advanced generation. A local offline synthesis engine runs automatically if key is omitted."
    )
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        st.success("API Key Active! 🟢")
    else:
        st.info("Running in Offline/Local RAG Mode ⚪")

    st.divider()
    
    # Quick Navigation / Tool Drawer
    st.markdown("### 🛠️ Built-in Internship Tools")
    tool_view = st.radio(
        "Inspect Module Details:",
        ["Active Conversation", "Task 1: Knowledge Base Manager", "Task 2: Multimodal Lab", "Task 5: Sentiment Evaluation"],
        index=0
    )

llm_client = LLMClient(api_key=user_api_key)
multimodal_analyzer = MultimodalAnalyzer(api_key=user_api_key)
image_generator = ImageGenerator(api_key=user_api_key)

# ==============================================================================
# VIEW: KNOWLEDGE BASE MANAGER (TASK 1)
# ==============================================================================
if tool_view == "Task 1: Knowledge Base Manager":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Task 1: Dynamic Knowledge Base Manager</h1>
        <div class="hero-subtitle">Real-time document ingestion, SHA-256 change tracking, chunking, and vector indexing</div>
    </div>
    """, unsafe_allow_html=True)

    stats = kb_manager.get_stats()
    
    # Metric Counters
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Documents Indexed", stats["total_documents"])
    with c2:
        st.metric("Total Chunks in Vector Store", stats["total_chunks"])
    with c3:
        st.metric("Last Dynamic Update", stats["last_updated"])

    st.markdown("### 📁 Currently Indexed Sources")
    sources = stats["indexed_sources"]
    if sources:
        st.write(", ".join([f"`{s}`" for s in sources]))
    else:
        st.warning("No sources currently indexed.")

    st.divider()

    # Interactive Update & Document Ingestion
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🔄 Trigger Dynamic Sync")
        st.write("Scans `data/customer_service/` for new, modified, or deleted documents and updates the vector database.")
        if st.button("🚀 Update Knowledge Base Now", use_container_width=True):
            with st.spinner("Scanning sources and re-indexing..."):
                res = kb_manager.update_knowledge_base(force_reload=True)
                st.success(res["message"])
                st.rerun()

    with col_b:
        st.markdown("#### ➕ Add New Knowledge Document")
        new_doc_title = st.text_input("Document Name (e.g., holiday_promotions.txt):")
        new_doc_text = st.text_area("Document Content:", height=150, placeholder="Paste new policy, product details, or FAQ here...")
        if st.button("📥 Ingest & Index New Document", use_container_width=True):
            if new_doc_title and new_doc_text.strip():
                with st.spinner("Ingesting new knowledge into vector store..."):
                    res = kb_manager.add_document_from_text(new_doc_title, new_doc_text)
                    st.success(f"Added `{new_doc_title}`! Index now contains {res['total_chunks']} chunks.")
                    st.rerun()
            else:
                st.error("Please provide both a document name and text content.")

    st.divider()
    st.markdown("#### 🔍 Test Knowledge Base Retrieval")
    kb_query = st.text_input("Test Query:", "What is the return policy window?")
    if kb_query:
        matches = kb_manager.query(kb_query, top_k=3)
        st.write(f"Found **{len(matches)}** matching chunks:")
        for idx, m in enumerate(matches):
            with st.expander(f"Rank {idx+1}: {m.get('source', 'Unknown')} (Similarity Score: {m.get('score', 0):.3f})"):
                st.write(m.get("text", ""))

# ==============================================================================
# VIEW: MULTIMODAL LAB (TASK 2)
# ==============================================================================
elif tool_view == "Task 2: Multimodal Lab":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Task 2: Multimodal Chatbot & Vision Lab</h1>
        <div class="hero-subtitle">Visual inspection for damaged products (Image-to-Text) and Google Imagen-3 Concept Generation (Text-to-Image)</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📸 Image Analysis (Vision)", "🎨 Image Generation (Imagen 3)"])

    with tab1:
        st.markdown("### 📸 Product Visual Inspection & Defect Analysis")
        st.write("Upload an image of a damaged product, shipping package, or receipt for diagnostic assessment.")
        uploaded_file = st.file_uploader("Upload Product Image:", type=["jpg", "jpeg", "png", "webp"])
        vision_prompt = st.text_input(
            "Vision Inspection Prompt:",
            "What problem can you identify with this product and how should customer service resolve it?"
        )

        if uploaded_file:
            img = Image.open(uploaded_file)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(img, caption="Uploaded Customer Image", use_container_width=True)
            with c2:
                if st.button("🔍 Run Multimodal Inspection", use_container_width=True):
                    with st.spinner("Analyzing image features and condition..."):
                        report = multimodal_analyzer.analyze_image(img, vision_prompt)
                        st.markdown(report["text"])
                        st.caption(f"Engine: {report['engine']} | Specifications: {report['image_meta']}")

    with tab2:
        st.markdown("### 🎨 Text-to-Image Generation")
        st.write("Generate visual product mockups, replacement previews, or illustrations using Google Imagen-3.")
        gen_prompt = st.text_area("Visual Prompt:", "A futuristic sleek wireless noise cancelling headphones on a clean wooden desk")
        if st.button("✨ Generate Concept Image", use_container_width=True):
            with st.spinner("Synthesizing visual content..."):
                gen_res = image_generator.generate_image(gen_prompt)
                st.image(gen_res["image"], caption=gen_res["message"], use_container_width=True)
                st.caption(f"Engine: {gen_res['engine']}")

# ==============================================================================
# VIEW: SENTIMENT EVALUATION (TASK 5)
# ==============================================================================
elif tool_view == "Task 5: Sentiment Evaluation":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Task 5: Sentiment Analysis & Quantitative Evaluation</h1>
        <div class="hero-subtitle">Real-time emotion detection, empathetic de-escalation, and benchmark dataset metrics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Benchmark Evaluation on `data/sentiment_test.csv`")
    if st.button("▶️ Run Evaluation Metrics", use_container_width=True):
        with st.spinner("Evaluating sentiment analyzer on test split..."):
            eval_metrics = sentiment_analyzer.evaluate_dataset()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{eval_metrics['accuracy'] * 100:.1f}%")
            c2.metric("Precision (Macro)", f"{eval_metrics['precision_macro'] * 100:.1f}%")
            c3.metric("Recall (Macro)", f"{eval_metrics['recall_macro'] * 100:.1f}%")
            c4.metric("F1-Score (Macro)", f"{eval_metrics['f1_macro'] * 100:.1f}%")

            st.divider()
            st.markdown("#### 🎯 Confusion Matrix")
            cm_df = pd.DataFrame(
                eval_metrics["confusion_matrix"],
                index=[f"Actual {l.title()}" for l in eval_metrics["labels"]],
                columns=[f"Predicted {l.title()}" for l in eval_metrics["labels"]]
            )
            st.dataframe(cm_df, use_container_width=True)

            st.divider()
            st.markdown("#### 📑 Per-Sample Predictions")
            details_df = pd.DataFrame(eval_metrics["details"])
            st.dataframe(details_df, use_container_width=True)

# ==============================================================================
# VIEW: ACTIVE CONVERSATION (MAIN UNIFIED CHATBOT)
# ==============================================================================
else:
    # --------------------------------------------------------------------------
    # MODE 1: CUSTOMER SERVICE (Core + Tasks 1, 2, 5, 6)
    # --------------------------------------------------------------------------
    if mode == "Customer Service":
        st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">ApexTech Customer Service — Extended</h1>
            <div class="hero-subtitle">GenAI Assistant with Dynamic Knowledge Retrieval, Sentiment Detection & Multilingual Support</div>
        </div>
        """, unsafe_allow_html=True)

        # Image attachment option for Customer Service (Task 2)
        with st.expander("📎 Optional: Attach Photo of Product / Damage for Inspection", expanded=False):
            cs_uploaded_image = st.file_uploader("Upload product photo:", type=["jpg", "png", "jpeg"], key="cs_uploader")
            if cs_uploaded_image:
                st.image(Image.open(cs_uploaded_image), width=200, caption="Attached Image for Chat Query")

        # Display Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                # Show metadata badges if available
                if "sentiment" in msg:
                    s_label = msg["sentiment"]["label"]
                    s_badge_class = f"badge-{s_label}"
                    st.markdown(f"""
                    <span class="badge {s_badge_class}">Sentiment: {s_label.upper()} ({msg['sentiment']['compound']})</span>
                    <span class="badge badge-lang">Language: {msg['lang']['display']}</span>
                    """, unsafe_allow_html=True)
                st.markdown(msg["content"])
                if "references" in msg and msg["references"]:
                    with st.expander(f"📚 Retrieved Reference Chunks ({len(msg['references'])})"):
                        for ref in msg["references"]:
                            st.caption(f"**Source: {ref.get('source', 'Unknown')}** (Relevance: {ref.get('score', 0):.2f})")
                            st.write(ref.get("text", ""))

        # Chat Input
        if user_prompt := st.chat_input("How can we help with your order, return, or warranty?"):
            # Step 1: Language Detection (Task 6)
            detected_lang = LanguageDetector.detect_language(user_prompt)
            active_lang_code = manual_lang_code or detected_lang["code"]
            active_lang_name = LanguageDetector.SUPPORTED_LANGUAGES.get(active_lang_code, "English")

            # Step 2: Sentiment Analysis (Task 5)
            sentiment_info = sentiment_analyzer.analyze(user_prompt)

            # Step 3: Knowledge Base Retrieval (Task 1)
            retrieved_chunks = kb_manager.query(user_prompt, top_k=3)
            context_text = "\n\n".join([f"[{c.get('source')}]: {c.get('text')}" for c in retrieved_chunks])

            # Append user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt,
                "sentiment": sentiment_info,
                "lang": detected_lang
            })

            with st.chat_message("user"):
                s_label = sentiment_info["label"]
                st.markdown(f"""
                <span class="badge badge-{s_label}">Sentiment: {s_label.upper()} ({sentiment_info['compound']})</span>
                <span class="badge badge-lang">Language: {detected_lang['display']}</span>
                """, unsafe_allow_html=True)
                st.markdown(user_prompt)

            # Step 4: Multimodal analysis if image was attached (Task 2)
            image_context = ""
            if cs_uploaded_image:
                img_obj = Image.open(cs_uploaded_image)
                vision_res = multimodal_analyzer.analyze_image(img_obj, user_prompt)
                image_context = f"\n[Multimodal Vision Inspection]: {vision_res['text']}"

            # Step 5: Construct System Instruction with Empathetic De-escalation
            system_instruction = f"""You are ApexTech's senior customer service AI.
{sentiment_info['tone_guidance']}
Respond in {active_lang_name}.
{image_context}
"""
            # Prepend empathy greeting if sentiment is negative
            empathy_prefix = ""
            if sentiment_info["label"] == "negative":
                empathy_prefix = MultilingualHandler.get_empathy_prefix(active_lang_code)

            # Step 6: Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Processing inquiry and retrieving knowledge..."):
                    llm_res = llm_client.generate_response(
                        prompt=user_prompt,
                        system_instruction=system_instruction,
                        conversation_history=st.session_state.messages,
                        retrieved_context=context_text,
                        target_language=active_lang_name
                    )
                    final_reply = empathy_prefix + llm_res["text"]
                    st.markdown(final_reply)

                    if retrieved_chunks:
                        with st.expander(f"📚 Retrieved Reference Chunks ({len(retrieved_chunks)})"):
                            for ref in retrieved_chunks:
                                st.caption(f"**Source: {ref.get('source', 'Unknown')}** (Score: {ref.get('score', 0):.2f})")
                                st.write(ref.get("text", ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_reply,
                "references": retrieved_chunks
            })

    # --------------------------------------------------------------------------
    # MODE 2: MEDICAL Q&A USING MEDQUAD (TASK 3)
    # --------------------------------------------------------------------------
    elif mode == "Medical Q&A (MedQuAD)":
        st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">Medical Q&A Assistant — MedQuAD Grounded</h1>
            <div class="hero-subtitle">NIH MedQuAD Question-Answer Retrieval with Clinical Entity Recognition</div>
        </div>
        <div class="disclaimer-box">
            <b>⚠️ MANDATORY EDUCATIONAL DISCLAIMER:</b> This tool provides educational information based strictly on the official MedQuAD dataset and is <b>NOT</b> a substitute for professional medical advice, clinical diagnosis, or emergency care.
        </div>
        """, unsafe_allow_html=True)

        # Quick Example Prompts
        st.markdown("##### 💡 Suggested MedQuAD Inquiries:")
        example_cols = st.columns(4)
        ex1 = example_cols[0].button("What is Acromegaly?")
        ex2 = example_cols[1].button("Symptoms of Addison's Disease")
        ex3 = example_cols[2].button("Type 2 Diabetes Prevention")
        ex4 = example_cols[3].button("Asthma Treatment Options")

        # History display
        for msg in st.session_state.medical_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "entities" in msg and msg["entities"]:
                    st.info(f"**Detected Medical Entities:**\n\n{MedicalEntityExtractor.format_entities_for_display(msg['entities'])}")
                if "retrieved" in msg and msg["retrieved"]:
                    with st.expander("🏥 Retrieved MedQuAD Reference"):
                        for r in msg["retrieved"]:
                            st.markdown(f"**Focus:** {r.get('focus')} | **QType:** {r.get('qtype')} | **Relevance:** {r.get('score', 0):.2f}")
                            st.write(r.get("text", ""))

        med_query = None
        if ex1: med_query = "What is Acromegaly and what causes it?"
        elif ex2: med_query = "What are the common symptoms of Addison's disease?"
        elif ex3: med_query = "How can I lower my risk of getting Type 2 Diabetes?"
        elif ex4: med_query = "What treatments and medications are used for asthma?"

        input_prompt = st.chat_input("Ask a medical inquiry (e.g. 'What are the symptoms of acromegaly?')...")
        if input_prompt:
            med_query = input_prompt

        if med_query:
            # Language Detection
            det_lang = LanguageDetector.detect_language(med_query)
            active_lang_name = LanguageDetector.SUPPORTED_LANGUAGES.get(manual_lang_code or det_lang["code"], "English")

            # Entity Extraction (Task 3)
            entities = MedicalEntityExtractor.extract_entities(med_query)

            # Retrieval from MedQuAD
            retrieved_med = medical_retriever.retrieve(med_query, top_k=2)
            med_context = ""
            for item in retrieved_med:
                med_context += f"Focus: {item.get('focus')}\nQuestion: {item.get('question')}\nAnswer: {item.get('answer')}\n\n"

            st.session_state.medical_messages.append({
                "role": "user",
                "content": med_query
            })

            with st.chat_message("user"):
                st.markdown(med_query)

            with st.chat_message("assistant"):
                # Display extracted entities
                st.markdown(f"**Detected Medical Entities:**\n\n{MedicalEntityExtractor.format_entities_for_display(entities)}")
                
                with st.spinner("Retrieving verified NIH MedQuAD knowledge..."):
                    sys_inst = (
                        "You are an educational medical knowledge assistant grounded exclusively in MedQuAD. "
                        "Cite symptoms, causes, or treatments directly from the provided text. "
                        "Include the educational disclaimer at the conclusion."
                    )
                    med_resp = llm_client.generate_response(
                        prompt=med_query,
                        system_instruction=sys_inst,
                        conversation_history=st.session_state.medical_messages,
                        retrieved_context=med_context,
                        target_language=active_lang_name
                    )
                    st.markdown(med_resp["text"])
                    
                    if retrieved_med:
                        with st.expander("🏥 Retrieved MedQuAD Reference"):
                            for r in retrieved_med:
                                st.markdown(f"**Focus:** {r.get('focus')} | **QType:** {r.get('qtype')} | **Relevance:** {r.get('score', 0):.2f}")
                                st.write(r.get("text", ""))

            st.session_state.medical_messages.append({
                "role": "assistant",
                "content": med_resp["text"],
                "entities": entities,
                "retrieved": retrieved_med
            })

    # --------------------------------------------------------------------------
    # MODE 3: RESEARCH EXPERT USING arXiv (TASK 4)
    # --------------------------------------------------------------------------
    elif mode == "Research Expert (arXiv)":
        st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">Scientific Expert Chatbot — arXiv Computer Science</h1>
            <div class="hero-subtitle">Semantic Paper Search, Structured Summarization, Mathematical Explanations & Visualizations</div>
        </div>
        """, unsafe_allow_html=True)

        res_tab1, res_tab2, res_tab3 = st.tabs(["💬 Research Chat & Explainer", "📄 Paper Summarizer", "📊 Concept & Paper Visualizations"])

        with res_tab1:
            st.markdown("##### 💡 Suggested Research Inquiries:")
            r_c1, r_c2, r_c3 = st.columns(3)
            rq1 = r_c1.button("Explain self-attention simply")
            rq2 = r_c2.button("Explain self-attention mathematically")
            rq3 = r_c3.button("Find papers about Vision Transformers")

            for msg in st.session_state.research_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "papers" in msg and msg["papers"]:
                        with st.expander(f"📚 Related arXiv Papers ({len(msg['papers'])})"):
                            for p in msg["papers"]:
                                st.markdown(f"**[{p.get('title')}]({p.get('url')})** (arXiv:{p.get('chunk_id')})")
                                st.caption(f"Authors: {p.get('authors')}")
                                st.write(p.get("abstract", ""))

            user_res_query = None
            if rq1: user_res_query = "Explain self-attention simply."
            elif rq2: user_res_query = "Explain self-attention mathematically with formulas."
            elif rq3: user_res_query = "Find papers about transformer architectures for image classification."

            res_input = st.chat_input("Ask a scientific question or search for CS papers...")
            if res_input: user_res_query = res_input

            if user_res_query:
                # Semantic Paper Search
                matched_papers = arxiv_searcher.search_papers(user_res_query, top_k=3)
                papers_context = ""
                for p in matched_papers:
                    papers_context += f"Paper Title: {p.get('title')}\nAuthors: {p.get('authors')}\nAbstract: {p.get('abstract')}\n\n"

                st.session_state.research_messages.append({
                    "role": "user",
                    "content": user_res_query
                })

                with st.chat_message("user"):
                    st.markdown(user_res_query)

                with st.chat_message("assistant"):
                    with st.spinner("Searching arXiv index and generating explanation..."):
                        sys_inst = (
                            "You are an expert AI research scientist assisting with Computer Science papers. "
                            "Explain concepts clearly, cite paper titles/authors where relevant, "
                            "and support mathematical depth if requested."
                        )
                        res_reply = llm_client.generate_response(
                            prompt=user_res_query,
                            system_instruction=sys_inst,
                            conversation_history=st.session_state.research_messages,
                            retrieved_context=papers_context,
                            target_language="English"
                        )
                        st.markdown(res_reply["text"])

                        if matched_papers:
                            with st.expander(f"📚 Related arXiv Papers ({len(matched_papers)})"):
                                for p in matched_papers:
                                    st.markdown(f"**[{p.get('title')}]({p.get('url')})** (arXiv:{p.get('chunk_id')})")
                                    st.caption(f"Authors: {p.get('authors')}")
                                    st.write(p.get("abstract", ""))

                st.session_state.research_messages.append({
                    "role": "assistant",
                    "content": res_reply["text"],
                    "papers": matched_papers
                })

        with res_tab2:
            st.markdown("### 📄 Structured Research Paper Summarizer")
            all_papers = arxiv_searcher.get_all_papers()
            paper_titles = [f"{p['id']} - {p['title']}" for p in all_papers]
            selected_idx = st.selectbox("Choose a paper from the arXiv CS dataset:", range(len(paper_titles)), format_func=lambda x: paper_titles[x])
            
            selected_paper = all_papers[selected_idx]
            structured_summary = PaperSummarizer.structure_summary_from_abstract(
                title=selected_paper["title"],
                abstract=selected_paper["abstract"],
                authors=selected_paper["authors_display"]
            )
            st.markdown(PaperSummarizer.format_markdown_summary(structured_summary))
            st.link_button("🔗 View Original on arXiv", selected_paper["url"])

        with res_tab3:
            st.markdown("### 📊 Interactive Visualizations")
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.plotly_chart(ResearchVisualizer.create_paper_topic_graph(arxiv_searcher.get_all_papers()), use_container_width=True)
            with v_col2:
                st.plotly_chart(ResearchVisualizer.create_concept_graph(), use_container_width=True)
