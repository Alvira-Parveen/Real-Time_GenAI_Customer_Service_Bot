import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Any

class ResearchVisualizer:
    """Generates interactive Plotly and NetworkX visualizations for papers and AI concepts."""

    @classmethod
    def create_paper_topic_graph(cls, papers: List[Dict[str, Any]]) -> go.Figure:
        """Builds a bipartite network connecting arXiv research papers to CS primary categories."""
        G = nx.Graph()

        # Category colors
        category_colors = {
            "cs.AI": "#FF5733", # Coral
            "cs.LG": "#33C1FF", # Sky Blue
            "cs.CL": "#33FF57", # Emerald
            "cs.CV": "#FF33F6"  # Magenta
        }

        # Add nodes
        for p in papers[:20]: # Limit for clear visualization
            paper_id = p["id"]
            short_title = p["title"][:28] + "..." if len(p["title"]) > 28 else p["title"]
            G.add_node(paper_id, node_type="paper", label=short_title, full_title=p["title"], authors=p.get("authors_display", ""))

            for cat in p.get("categories", []):
                if not G.has_node(cat):
                    G.add_node(cat, node_type="category", label=cat, full_title=f"Category: {cat}", authors="")
                G.add_edge(paper_id, cat)

        # Spring layout for balanced graph aesthetics
        pos = nx.spring_layout(G, k=0.45, iterations=40, seed=42)

        # Extract edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color='#888888'),
            hoverinfo='none',
            mode='lines'
        )

        # Extract nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            info = G.nodes[node]
            
            if info["node_type"] == "category":
                node_size.append(28)
                color = category_colors.get(node, "#FFAA00")
                node_color.append(color)
                node_text.append(f"<b>Category: {node}</b><br>Papers linked: {G.degree(node)}")
            else:
                node_size.append(14)
                node_color.append("#4A90E2")
                node_text.append(f"<b>{info['full_title']}</b><br>ID: arXiv:{node}<br>Authors: {info['authors']}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[n]["label"] if G.nodes[n]["node_type"] == "category" else "" for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='#FFFFFF')
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<b>arXiv Research Landscape: Papers & Topic Clusters</b>",
                title_x=0.5,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=10, r=10, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(240,242,246,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=500
            )
        )
        return fig

    @classmethod
    def create_concept_graph(cls) -> go.Figure:
        """Visualizes conceptual relationships in modern Deep Learning architectures."""
        concepts = {
            "Self-Attention": ["Transformers", "Query-Key-Value", "Scaled Dot-Product", "Multi-Head Attention"],
            "Transformers": ["BERT", "GPT-4", "Vision Transformer", "LoRA", "RAG"],
            "Vision Transformer": ["Patch Embeddings", "ImageNet Benchmark", "Self-Attention"],
            "LoRA": ["Parameter-Efficient Fine-Tuning", "Rank Decomposition"],
            "RAG": ["Vector Embeddings", "Dense Retrieval", "Hallucination Mitigation"],
            "GPT-4": ["Instruction Tuning", "RLHF", "Few-Shot Learning"]
        }

        G = nx.DiGraph()
        for parent, children in concepts.items():
            for child in children:
                G.add_edge(parent, child)

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=10)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#99AAB5'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_labels = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="bottom center",
            hoverinfo='text',
            hovertext=[f"<b>Concept:</b> {n}" for n in node_labels],
            marker=dict(
                size=18,
                color='#7289DA',
                line=dict(width=2, color='#2C2F33')
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<b>Deep Learning & NLP Concept Dependency Map</b>",
                title_x=0.5,
                showlegend=False,
                margin=dict(b=20, l=10, r=10, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(240,242,246,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=450
            )
        )
        return fig
