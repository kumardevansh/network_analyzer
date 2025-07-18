import os
import tempfile
import networkx as nx
import sympy as sp
import re
from collections import defaultdict
import gradio as gr
from gradio.themes import Ocean
from huggingface_hub import InferenceClient
import requests
from PIL import Image
from io import BytesIO

# --- Set your HF token ---
# Either hardcode here or use Colab's userdata like:
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not set. Please add it in your Space's secrets.")

from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_TOKEN"],
)

# --- Helper: Save PIL image to URL-accessible temp file ---
def image_to_temp_url(image):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_path.name)
    return "https://your-server.com/temporary-image-support.png"  # placeholder (host image externally if needed)

# --- OR upload image to Hugging Face Space / GDrive and return a public URL instead
# You can use this for production use

def extract_network_from_image(image):
    # Upload image to temp path
    image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    image.save(image_path)

    # Upload manually or serve image online if needed
    # For now, simulate by loading image into bytes and re-uploading to HF or GDrive
    # Instead: In Colab, just use direct GDrive URLs

    # Placeholder: Manually put a URL here for now (from GDrive or HF Spaces or web)
    raise NotImplementedError("Replace this with your public image URL logic.")

# New: Directly send the URL to Unsloth Mistral + get output
def extract_network_from_url(image_url):
    prompt = (
        "Analyze this network diagram and list the network only, e.g. Q + W -> R. Do not print any other sentence except the network."
        "The arrows represent reactions. If there are multiple reactions, give them comma separated like A -> B, B -> C, etc."
    )

    completion = client.chat.completions.create(
        model="unsloth/Mistral-Small-3.2-24B-Instruct-2506",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            }
        ]
    )

    return completion.choices[0].message.content.strip()


# --- Network Analysis Functions ---
def parse_species(expr):
    return [s.strip() for s in re.split(r'\s*[\+\-]\s*', expr)]

def parse_network(input_string):
    edges, reversible_edges = [], []
    for part in input_string.split(','):
        part = part.strip()
        if '<->' in part:
            lhs, rhs = part.split('<->')
            lhs_species = parse_species(lhs)
            rhs_species = parse_species(rhs)
            reversible_edges.append((lhs_species, rhs_species))
        elif '->' in part:
            lhs, rhs = part.split('->')
            lhs_species = parse_species(lhs)
            rhs_species = parse_species(rhs)
            edges.append((lhs_species, rhs_species))
    return edges, reversible_edges

def build_graph(edges, reversible_edges):
    G = nx.DiGraph()
    for a, b in edges:
        lhs = " + ".join(a)
        rhs = " + ".join(b)
        G.add_edge(lhs, rhs)
    for a, b in reversible_edges:
        lhs = " + ".join(a)
        rhs = " + ".join(b)
        G.add_edge(lhs, rhs)
        G.add_edge(rhs, lhs)
    return G

def analyze_graph(G):
    return {
        "nodes": list(G.nodes),
        "edges": list(G.edges),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_cyclic": not nx.is_directed_acyclic_graph(G)
    }

def mass_action_odes(edges, reversible_edges):
    species = set()
    odes = defaultdict(lambda: 0)
    rate_counter = 1

    def term(species_list):
        term_expr = 1
        for s in species_list:
            sym = sp.symbols(s)
            species.add(sym)
            term_expr *= sym
        return term_expr

    for lhs_species, rhs_species in edges:
        k = sp.symbols(f'k{rate_counter}')
        rate_counter += 1
        flux = k * term(lhs_species)
        for s in lhs_species:
            sym = sp.symbols(s)
            odes[sym] -= flux
        for s in rhs_species:
            sym = sp.symbols(s)
            odes[sym] += flux

    for lhs_species, rhs_species in reversible_edges:
        kf = sp.symbols(f'k{rate_counter}')
        rate_counter += 1
        kr = sp.symbols(f'k{rate_counter}')
        rate_counter += 1
        forward_flux = kf * term(lhs_species)
        reverse_flux = kr * term(rhs_species)
        for s in lhs_species:
            sym = sp.symbols(s)
            odes[sym] -= forward_flux
            odes[sym] += reverse_flux
        for s in rhs_species:
            sym = sp.symbols(s)
            odes[sym] += forward_flux
            odes[sym] -= reverse_flux

    return dict(odes)

def format_odes(odes):
    return "\n".join([f"d{var}/dt = {sp.simplify(expr)}" for var, expr in odes.items()])

def compute_jacobian(odes):
    variables = list(odes.keys())
    F = sp.Matrix([odes[var] for var in variables])
    J = F.jacobian(variables)
    return sp.pretty(J)

def process_network(input_string, query, image_url=None):
    edges, reversible_edges = parse_network(input_string)
    G = build_graph(edges, reversible_edges)
    info = analyze_graph(G)

    if 'ode' in query.lower():
        ode_sys = mass_action_odes(edges, reversible_edges)
        return format_odes(ode_sys)
    elif 'jacobian' in query.lower():
        ode_sys = mass_action_odes(edges, reversible_edges)
        return f"Jacobian Matrix:\n{compute_jacobian(ode_sys)}"
    elif 'variables' in query.lower():
        return f"There are {info['num_nodes']} variables: {info['nodes']}"
    elif 'edges' in query.lower():
        return f"Edges: {info['edges']}"
    elif 'cyclic' in query.lower() or 'cycle' in query.lower():
        cycles = list(nx.simple_cycles(G))
        return "Cycles found:\n" + "\n".join([" -> ".join(cycle + [cycle[0]]) for cycle in cycles]) if cycles else "No cycles found."

    # Fallback: Use LLM on both image and parsed network
    else:
        content = [
            {
                "type": "text",
                "text": (
                    "You are given a biological network with the following structure:\n"
                    f"• Nodes: {info['nodes']}\n"
                    f"• Reactions (edges): {info['edges']}\n\n"
                    f"Answer the following query based on this structure and the image:"
                    f"\n\n{query}"
                ),
            }
        ]
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        response = client.chat.completions.create(
            model="unsloth/Mistral-Small-3.2-24B-Instruct-2506",
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content.strip()

# --- Full Gradio Handler ---
def full_process(text_input, image_url, query):
    image_preview = None
    network_description = ""
    result = ""

    if text_input.strip():
        network_description = text_input.strip()
    elif image_url.strip():
        # Display image from URL
        try:
            response = requests.get(image_url)
            image_preview = Image.open(BytesIO(response.content))
        except:
            return None, "", "❌ Invalid image URL"

        # Extract network
        network_description = extract_network_from_url(image_url)
    else:
        return None, "", "❌ Provide text or image URL."

    # Answer query
    result = process_network(network_description, query, image_url=image_url if image_url.strip() else None)
    return image_preview, network_description, result

import gradio as gr
from gradio.themes.utils import sizes
from gradio.themes.base import Base
from gradio.themes.utils import colors

# Optional: Keep your theme
theme = gr.themes.Ocean()

with gr.Blocks(theme=theme, css="#footer-link {text-align: center; font-size: 14px; color: #555;}") as iface:

    gr.Markdown("## 🔬 Biological Network Analyzer (Multimodal Mistral via Unsloth)")
    gr.Markdown("Paste a network OR provide a public image URL. Then ask a query like **'Give ODEs'** or **'Is it cyclic?'**")

    with gr.Row():
        with gr.Column():
            # img_input = gr.Image(type="pil", label="Upload Network Image (❌ Not supported unless image is hosted online)")
            text_input = gr.Textbox(label="Text Input (optional)", placeholder="Or paste network: A + B -> C, X <-> Y")
            url_input = gr.Textbox(label="🔗 Public Image URL (e.g., from GDrive)", placeholder="https://... (must be accessible)")
            query_input = gr.Textbox(label="Query", placeholder="Ask about ODEs, Jacobian, edges, etc.")

        with gr.Column():
            img_output = gr.Image(label="🖼️ Image Preview")
            network_text = gr.Textbox(label="🧪 Extracted Network")
            result_box = gr.Textbox(label="📘 Answer")




    # Link logic to function
    inputs = [text_input, url_input, query_input]
    outputs = [img_output, network_text, result_box]
    iface_fn = gr.Interface(fn=full_process, inputs=inputs, outputs=outputs)

    # Footer GitHub link
    gr.Markdown("""
      <footer style='text-align:center; margin-top:20px; color:#aaa;'>
          Built using Gradio, Hugging Face & Mistral |
          <a href="https://github.com/kumardevansh/network_analyzer" target="_blank" style="color:#aaa; text-decoration:underline;">
              View on GitHub
          </a>
      </footer>
    """)


iface.launch(share=True)
