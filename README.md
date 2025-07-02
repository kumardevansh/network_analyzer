
# 🧬 Biological Network Analyzer

An intelligent assistant for analyzing biological networks from **text or image input**. This app can extract reaction networks from uploaded diagrams using **Gemini Vision API**, parse them into graph structures, and generate:

- 📈 ODEs (Ordinary Differential Equations)
- 🧮 Jacobian matrices
- 🔄 Cyclic dependency checks
- 📊 Graph stats (edges, variables)

All via a **Gradio-based conversational interface**.

---

## 🚀 Features

- 🔤 **Text input** (e.g. `A + B -> C, C <-> D`)
- 🖼️ **Image input** (upload scanned or digital diagrams)
- 🧠 Google Gemini API integration to extract text-based reactions from images
- 🧪 ODE system generation using mass-action kinetics
- 🧠 Jacobian matrix computation via symbolic algebra (SymPy)
- 🔁 Cycle detection and variable/edge counting
- 🤖 LLM-based fallback Q&A via `flan-t5-base`
- 🎨 Customizable Gradio theme (e.g. `Soft`, `Glass`, etc.)

---

## 🧰 Tech Stack

| Component     | Technology                     |
|---------------|-------------------------------|
| Frontend UI   | [Gradio](https://gradio.app)  |
| LLM Query     | [Flan-T5](https://huggingface.co/google/flan-t5-base) |
| Image-to-Text | [Gemini API](https://ai.google.dev/) (Vision multimodal API) |
| Graph Parsing | [NetworkX](https://networkx.org) |
| Symbolic Math | [SymPy](https://www.sympy.org/) |

---

## 🖥️ Running the App

### ✅ Prerequisites

- Python ≥ 3.8
- Install dependencies:

```bash
pip install gradio transformers networkx sympy google-generativeai python-dotenv pillow
````

* Set your Gemini API key in a `.env` file:

```
GEMINI_API=your_api_key_here
```

### ▶️ Run

```bash
python your_script.py
```

Or use inside a [Google Colab notebook](https://colab.research.google.com/) for instant testing.

---

## 📷 Example Inputs

### Text:

```
A + B -> C, C <-> D, D -> E
```

### Image:

* Upload PNG or JPG of a biological reaction network
* Gemini will parse it automatically

---

## 📌 Future Additions

* Export ODEs/Jacobian as LaTeX/PDF
* Stoichiometry & reaction rate customization

---

## 🧑‍🔬 Author

Built by \[Devansh] for a project on **LLM-based Equation Discovery from Biological Networks**.

---

## 📝 License

MIT License
