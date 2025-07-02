from transformers import pipeline
import gradio as gr
import networkx as nx
import sympy as sp
from collections import defaultdict
import re
from dotenv import load_dotenv
import google.generativeai as genai
import os
from gradio.themes import Ocean

load_dotenv()
API_KEY = os.getenv("GEMINI_API")
genai.configure(api_key=API_KEY)

# Initialize the Gemini Flash Model
model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')


# Gradio App with Support for Multi-Reactant Networks (e.g. A + B -> AB)

# --- Parsing Functions ---
def parse_species(expr):
    # e.g., "A + B" -> ["A", "B"]
    return [s.strip() for s in re.split(r'\s*[\+\-]\s*', expr)]

def parse_network(input_string):
    edges = []
    reversible_edges = []

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

# --- ODE Generator for Complex Reactions ---
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

def extract_network_from_image(image):
    prompt = (
        "Analyze this network diagram and list the network only. "
        "Use reaction format like 'A + B -> C' or 'X <-> Y'. "
        "List multiple reactions separated by commas."
    )
    gemini_response = model.generate_content([prompt, image])
    return gemini_response.text.strip()


def full_process(image, text_input, query):
    if text_input.strip():  # If text is given, use it
        network_description = text_input.strip()
    elif image is not None:  # Else if image is given, extract network from image
        network_description = extract_network_from_image(image)
    else:
        return "❌ Please provide either a network image or a textual description."

    # Step 2: Process extracted/generated network
    return process_network(network_description, query)


qa = pipeline("text2text-generation", model="google/flan-t5-base")
def process_network(input_string, query):
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

    elif 'cyclic' or 'cycle' in query.lower():
        cycles = list(nx.simple_cycles(G))
        if cycles:
            cycles_str = "\n".join([" -> ".join(cycle + [cycle[0]]) for cycle in cycles])
            return f"Cycles found:\n{cycles_str}"
        else:
            return "No cycles found."

    else:
        prompt = f"Given the network with nodes: {info['nodes']} and edges: {info['edges']}, answer the query: {query}"
        answer = qa(prompt, max_length=128)[0]['generated_text']
        return answer


iface = gr.Interface(
    fn=full_process,
    inputs=[
        gr.Image(type="pil", label="Upload Network Image (optional)"),
        gr.Textbox(label="Text Input (optional)", placeholder="Or paste network: A + B -> C, X <-> Y"),
        gr.Textbox(label="Query", placeholder="Ask about ODEs, Jacobian, edges, etc.")
    ],
    outputs="text",
    title="Biological Network Analyzer",
    description="Upload an image or enter network text. Then ask a query like 'Give ODEs' or 'Is it cyclic?'.",
    theme=Ocean()
)

iface.launch(share=True)
