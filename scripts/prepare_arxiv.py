import os
import requests
import json
import xml.etree.ElementTree as ET
import time

ARXIV_DIR = "datasets/arxiv"
os.makedirs(ARXIV_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(ARXIV_DIR, "arxiv_cs_papers.json")

# Seminal and landmark Computer Science papers (arXiv IDs, Titles, Authors, Abstracts, Categories)
CURATED_SEMINAL_PAPERS = [
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
        "categories": ["cs.CL", "cs.LG"],
        "published": "2017-06-12",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.",
        "url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
    },
    {
        "id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "categories": ["cs.CL"],
        "published": "2018-10-11",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.",
        "url": "https://arxiv.org/abs/1810.04805",
        "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf"
    },
    {
        "id": "2010.11929",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "authors": ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", "Dirk Weissenborn", "Xiaohua Zhai", "Thomas Unterthiner", "Mostafa Dehghani", "Matthias Minderer", "Georg Heigold", "Sylvain Gelly", "Jakob Uszkoreit", "Neil Houlsby"],
        "categories": ["cs.CV", "cs.AI", "cs.LG"],
        "published": "2020-10-22",
        "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.",
        "url": "https://arxiv.org/abs/2010.11929",
        "pdf_url": "https://arxiv.org/pdf/2010.11929.pdf"
    },
    {
        "id": "2005.14165",
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry", "Amanda Askell", "Sandhini Agarwal", "Ariel Herbert-Voss", "Gretchen Krueger", "Tom Henighan", "Rewon Child", "Aditya Ramesh", "Daniel M. Ziegler", "Jeffrey Wu", "Clemens Winter", "Christopher Hesse", "Mark Chen", "Eric Sigler", "Mateusz Litwin", "Scott Gray", "Benjamin Chess", "Jack Clark", "Christopher Berner", "Sam McCandlish", "Alec Radford", "Ilya Sutskever", "Dario Amodei"],
        "categories": ["cs.CL", "cs.AI", "cs.LG"],
        "published": "2020-05-28",
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. We train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting.",
        "url": "https://arxiv.org/abs/2005.14165",
        "pdf_url": "https://arxiv.org/pdf/2005.14165.pdf"
    },
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Küttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"],
        "categories": ["cs.CL", "cs.AI", "cs.LG"],
        "published": "2020-05-22",
        "abstract": "Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and they often hallucinate incorrect facts. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) - models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.",
        "url": "https://arxiv.org/abs/2005.11401",
        "pdf_url": "https://arxiv.org/pdf/2005.11401.pdf"
    },
    {
        "id": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
        "categories": ["cs.CL", "cs.AI", "cs.LG"],
        "published": "2021-06-17",
        "abstract": "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. LoRA can reduce the number of trainable parameters by 10,000 times and GPU memory requirements by 3 times compared to full fine-tuning.",
        "url": "https://arxiv.org/abs/2106.09685",
        "pdf_url": "https://arxiv.org/pdf/2106.09685.pdf"
    },
    {
        "id": "2305.18290",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"],
        "categories": ["cs.LG", "cs.AI", "cs.CL"],
        "published": "2023-05-29",
        "abstract": "While large-scale unsupervised language models learn broad world knowledge and reasoning capabilities, controlling their behavior precisely through reinforcement learning from human feedback (RLHF) has proven complex and unstable. We propose Direct Preference Optimization (DPO), an algorithm that implicitly optimizes the same objective as existing RLHF algorithms (reward maximization with a KL penalty) but is simple to train and eliminates the need to fit a separate reward model or sample from the LM during training.",
        "url": "https://arxiv.org/abs/2305.18290",
        "pdf_url": "https://arxiv.org/pdf/2305.18290.pdf"
    },
    {
        "id": "1512.03385",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        "categories": ["cs.CV"],
        "published": "2015-12-10",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those previously used. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity.",
        "url": "https://arxiv.org/abs/1512.03385",
        "pdf_url": "https://arxiv.org/pdf/1512.03385.pdf"
    },
    {
        "id": "2203.02155",
        "title": "Training language models to follow instructions with human feedback",
        "authors": ["Long Ouyang", "Jeff Wu", "Xu Jiang", "Diogo Almeida", "Carroll L. Wainwright", "Pamela Mishkin", "Chong Zhang", "Sandhini Agarwal", "Katarina Slama", "Alex Ray", "John Schulman", "Jacob Hilton", "Fraser Kelton", "Luke Miller", "Maddie Simens", "Amanda Askell", "Peter Welinder", "Paul Christiano", "Jan Leike", "Ryan Lowe"],
        "categories": ["cs.CL", "cs.AI", "cs.LG"],
        "published": "2022-03-04",
        "abstract": "Making language models larger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. We show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback (RLHF). We call the resulting models InstructGPT.",
        "url": "https://arxiv.org/abs/2203.02155",
        "pdf_url": "https://arxiv.org/pdf/2203.02155.pdf"
    },
    {
        "id": "2103.00020",
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "authors": ["Alec Radford", "Jong Wook Kim", "Chris Hallacy", "Aditya Ramesh", "Gabriel Goh", "Sandhini Agarwal", "Girish Sastry", "Amanda Askell", "Pamela Mishkin", "Jack Clark", "Gretchen Krueger", "Ilya Sutskever"],
        "categories": ["cs.CV", "cs.CL", "cs.LG"],
        "published": "2021-03-01",
        "abstract": "State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. The resulting model, CLIP (Contrastive Language-Image Pre-training), transfers readily to over 30 existing computer vision datasets with competitive zero-shot performance.",
        "url": "https://arxiv.org/abs/2103.00020",
        "pdf_url": "https://arxiv.org/pdf/2103.00020.pdf"
    },
    {
        "id": "2112.10752",
        "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
        "authors": ["Robin Rombach", "Andreas Blattmann", "Dominik Lorenz", "Patrick Esser", "Björn Ommer"],
        "categories": ["cs.CV", "cs.LG"],
        "published": "2021-12-20",
        "abstract": "By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data. Furthermore, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, operating directly in pixel space demands immense computational resources. We propose Latent Diffusion Models (LDMs), which apply diffusion in the latent space of pre-trained autoencoders, drastically reducing computational requirements while retaining perceptual quality and synthesis fidelity.",
        "url": "https://arxiv.org/abs/2112.10752",
        "pdf_url": "https://arxiv.org/pdf/2112.10752.pdf"
    },
    {
        "id": "2303.08774",
        "title": "GPT-4 Technical Report",
        "authors": ["OpenAI"],
        "categories": ["cs.CL", "cs.AI"],
        "published": "2023-03-15",
        "abstract": "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. We also outline how safety training and red-teaming were incorporated into model alignment.",
        "url": "https://arxiv.org/abs/2303.08774",
        "pdf_url": "https://arxiv.org/pdf/2303.08774.pdf"
    },
    {
        "id": "2307.09288",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": ["Hugo Touvron", "Louis Martin", "Kevin Stone", "Peter Albert", "Amjad Almahairi", "Yasmine Babaei", "Nikolay Bashlykov", "Soumya Batra", "Prajjwal Bhargava", "Shruti Bhosale", "Dan Bikel", "Lukas Blecher", "Cristian Canton Ferrer", "Moya Chen", "Guillem Cucurull", "David Esiobu", "Jude Fernandes", "Jeremy Fu", "Wenyin Fu", "Brian Fuller", "Cynthia Gao", "Vedanuj Goswami", "Naman Goyal", "Anthony Hartshorn", "Saghar Hosseini", "Rui Hou", "Hakan Inan", "Marcin Kardas", "Viktor Kerkez", "Madian Khabsa", "Isabel Kloumann", "Artem Korenev", "Punit Singh Koura", "Marie-Anne Lachaux", "Thibaut Lavril", "Jenya Lee", "Diana Liskovich", "Yinghai Lu", "Yuning Mao", "Xavier Martinet", "Todor Mihaylov", "Pushkar Mishra", "Igor Molybog", "Yixin Nie", "Andrew Poulton", "Jeremy Reizenstein", "Rashi Rungta", "Kalyan Saladi", "Alan Schelten", "Ruan Silva", "Eric Michael Smith", "Ranjan Subramanian", "Xiaoqing Ellen Tan", "Binh Tang", "Ross Taylor", "Adina Williams", "Jian Xiang Kuan", "Puxin Xu", "Zheng Yan", "Iliyan Zarov", "Yuchen Zhang", "Angela Fan", "Melanie Kambadur", "Sharan Narang", "Aurelien Rodriguez", "Robert Stojnic", "Sergey Edunov", "Thomas Scialom"],
        "categories": ["cs.CL", "cs.AI", "cs.LG"],
        "published": "2023-07-18",
        "abstract": "In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Across the benchmarks we tested, Llama 2-Chat models outperform open-source chat models, and based on human evaluations for helpfulness and safety, are on par with some popular closed-source models like ChatGPT and PaLM.",
        "url": "https://arxiv.org/abs/2307.09288",
        "pdf_url": "https://arxiv.org/pdf/2307.09288.pdf"
    },
    {
        "id": "1409.0473",
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "authors": ["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
        "categories": ["cs.CL", "cs.LG"],
        "published": "2014-09-01",
        "abstract": "Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder-decoders and encode a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word.",
        "url": "https://arxiv.org/abs/1409.0473",
        "pdf_url": "https://arxiv.org/pdf/1409.0473.pdf"
    },
    {
        "id": "2201.11903",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Ed Chi", "Quoc Le", "Denny Zhou"],
        "categories": ["cs.CL", "cs.AI"],
        "published": "2022-01-28",
        "abstract": "We explore how generating a chain of thought - a series of intermediate reasoning steps - significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on arithmetic, commonsense, and symbolic reasoning benchmarks show that chain-of-thought prompting outperforms standard prompting, achieving state-of-the-art accuracy on the GSM8K benchmark.",
        "url": "https://arxiv.org/abs/2201.11903",
        "pdf_url": "https://arxiv.org/pdf/2201.11903.pdf"
    }
]

def fetch_additional_arxiv(query="cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL", max_results=25):
    url = f"https://export.arxiv.org/api/query?search_query={requests.utils.quote(query)}&max_results={max_results}&sortBy=relevance"
    print(f"Querying arXiv API: {url}...")
    papers = []
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            # namespace
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", default="", namespaces=ns).replace("\n", " ").strip()
                summary = entry.findtext("atom:summary", default="", namespaces=ns).replace("\n", " ").strip()
                id_elem = entry.findtext("atom:id", default="", namespaces=ns)
                paper_id = id_elem.split("/abs/")[-1] if "/abs/" in id_elem else id_elem
                published = entry.findtext("atom:published", default="", namespaces=ns)[:10]
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.findtext("atom:name", default="", namespaces=ns)
                    if name:
                        authors.append(name.strip())
                        
                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.attrib.get("term", "")
                    if term and term.startswith("cs."):
                        categories.append(term)
                if not categories:
                    categories = ["cs.AI"]
                    
                papers.append({
                    "id": paper_id,
                    "title": title,
                    "authors": authors,
                    "categories": categories,
                    "published": published,
                    "abstract": summary,
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf"
                })
            print(f"Fetched {len(papers)} live papers from arXiv API.")
    except Exception as e:
        print(f"arXiv live fetch error or timeout: {e}")
        
    return papers

def main():
    combined = list(CURATED_SEMINAL_PAPERS)
    seen_ids = set(p["id"] for p in combined)
    
    live_papers = fetch_additional_arxiv(max_results=20)
    for p in live_papers:
        if p["id"] not in seen_ids:
            combined.append(p)
            seen_ids.add(p["id"])
            
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully prepared {len(combined)} Computer Science arXiv papers in {OUTPUT_FILE}!")

if __name__ == "__main__":
    main()
