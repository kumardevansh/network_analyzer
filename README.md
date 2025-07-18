# 🔬 Biological Network Analyzer (LLM + Vision + Symbolic Reasoning)

A multimodal tool to **analyze biological reaction networks** from either structured text or **public image URLs**. Combines **vision-language models** with symbolic computation to answer user queries like:

> *"Give the ODEs"*, *"Is the system cyclic?"*, *"Show the Jacobian"*, etc.

---

## 🧠 Features

* 🧾 Accepts **text-based** or **image-based** network inputs (e.g., `A + B -> C`, `X <-> Y`)
* 🧮 Computes **ODEs and Jacobians** via **mass-action kinetics**
* 🔁 Detects **cycles and feedback loops** in the network using **NetworkX**
* 🖼️ Uses **Unsloth Mistral 24B Vision LLM** via Hugging Face’s **InferenceClient** to extract networks from image URLs
* 💬 Interactive interface built with **Gradio + Ocean theme**

---

## 🚀 Demo Workflow

1. **Input** a network in text (`A + B -> C`) or **provide a public image URL** of a network diagram (e.g., from Google Drive).
2. **Ask queries** like:

   * “Give ODEs”
   * “Show Jacobian”
   * “List variables”
   * “Is it cyclic?”
3. **Output**: Computed answers and optional image preview.

---

## 🛠️ Tech Stack

| Domain               | Tools / Libraries                                 |
| -------------------- | ------------------------------------------------- |
| LLM & Vision         | `Unsloth/Mistral-24B-Instruct` (via Hugging Face) |
| Interface            | `Gradio`, `Ocean` theme                           |
| Graph Logic          | `NetworkX`, `re`, `defaultdict`                   |
| Symbolic Computation | `SymPy`                                           |
| Image Handling       | `PIL`, `requests`, `BytesIO`                      |
| Hosting (Planned)    | GDrive / Hugging Face Spaces (for image URLs)     |

---

## 📝 Sample Input

**Network (Text):**

```
A + B -> C, C <-> D, D -> A
```

**Query:**

```
Is it cyclic?
```

**Network (Image URL):**

```
https://drive.google.com/uc?export=download&id=1gCza0lEBlK9Ox88qLi4V1Yt1jIdnEp3Y
```

---

## ⚙️ How to Run

1. Clone the repo (or copy to Google Colab).
2. Set your Hugging Face API token via:

```python
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')  # or manually
```

3. Run the script:

```bash
python app.py  # or via Colab cells
```

---

## 🔒 Notes

* **Image Upload** from local disk is currently disabled. Only public image URLs are supported.
* Internally, images are passed to **Mistral VLM** for vision-text understanding to extract reactions.
* The logic currently follows **mass-action kinetics** assumptions.

---

## 📌 Example Use Case

> You upload a biological pathway image or describe it as `e2 + GDP <-> e2GDP, e2GDP + e5 <-> Ce1`.
> The app generates:

* ODEs for each species
* Jacobian matrix
* Graph structure and cycle detection
* Answer to general queries using the LLM

---
