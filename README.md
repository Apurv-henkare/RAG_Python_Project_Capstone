# 🎮 Love2D Game Engine AI Documentation Assistant (RAG)

An enterprise-ready **Retrieval-Augmented Generation (RAG)** application built with **LangChain**, **NVIDIA Embeddings**, **FAISS**, and **Google Gemini**. 

It transforms a library of 24 technical PDF manuals for the **LÖVE (Love2D)** game development engine into an interactive, hallucination-free AI assistant with both a **Web Chat UI** and a **Command Line Interface**.

---

## 📌 1. Project Overview & Use Case

### The Real-World Problem:
In the game development industry, studios frequently use **in-house, proprietary game engines** (like EA's Frostbite, Rockstar's RAGE, or Capcom's RE Engine). These engines have private documentation that is **not available on the public internet**. 
- Public AI tools like ChatGPT know nothing about internal engines and hallucinate incorrect code.
- New game developers take **3 to 6 months** to read hundreds of pages of complex manuals.

### The Solution:
This project acts as an **Internal Engine Knowledge Assistant**:
- New engineers can ask questions in plain English (e.g., *"How do I check bounding box collisions in Lua?"*).
- The system searches the studio's private PDF manuals in milliseconds and returns **exact, working code examples with explanations**.
- It includes strict **anti-hallucination guardrails** that politely refuse to answer when a question is outside the documentation (e.g., asking for C# or unrelated physics).

---

## 🎥 Video Demonstration & Walkthrough

A full video walkthrough showcasing the application, Streamlit web chat UI, CLI interface, and evaluation benchmark is available here:

👉 **[Watch Live Project Demonstration (Google Drive)](https://drive.google.com/file/d/1l4-MTH5MKmjcRnFYH5K3B23TyKG0WOT8/view?usp=sharing)**

---

## 🏗️ 2. System Architecture & Flow

The system follows a modern RAG pipeline utilizing LangChain Expression Language (LCEL) with modular components.

### Architecture Diagram:

```text
================================================================================================
                    PHASE 1: OFFLINE DOCUMENT INGESTION & VECTOR INDEXING
                                       (Runs Once)
================================================================================================

  [ 24 Love2D PDF Manuals ]
            │
            ▼
  [ pypdf Text Extraction ]       ──> Extracts text & attaches metadata (source filename, page)
            │
            ▼
  [ Recursive Character Splitter] ──> Chunks text (chunk_size: 3000, overlap: 600)
            │
            ▼
  [ NVIDIA Embedding Model ]      ──> Model: nvidia/nemotron-3-embed-1b (Converts text to vectors)
            │
            ▼
  [ Local FAISS Vector Store ]    ──> Cached to disk (faiss_index/) for sub-0.1s instant startup!

================================================================================================
                    PHASE 2: RUNTIME QUERY & GENERATION PIPELINE
                         (Streamlit Web UI / Terminal CLI)
================================================================================================

  [ User Question ]  ──> e.g., "How do I check for collisions in Lua?"
         │
         ├───► [ NVIDIA Embedding ] ──► Converts query to vector
         │                                      │
         │                                      ▼
         │                            [ FAISS Vector Search ]
         │                            Searches cached index & retrieves
         │                            Top 5 most relevant chunks (k=5)
         │                                      │
         │                                      ▼
         └────────────────────────────► [ LCEL Prompt Assembly ]
                                        Injects: Question + Retrieved Chunks
                                                │
                                                ▼
                                        [ Google Gemini LLM ]
                                        Model: gemini-3.1-flash-lite
                                        Synthesizes answer strictly from context
                                                │
                                                ▼
                                        [ Pydantic Output Parser ]
                                        Enforces JSON Schema: Transformer(text, context)
                                                │
                                                ▼
                                        [ Anti-Hallucination Guardrail ]
                                        Suppresses context if question is out-of-scope
                                                │
                                                ▼
                                 [ Clean UI Display with Lua Code ]
                                 Displays formatted answer + ```lua code block
================================================================================================
```

### End-to-End Pipeline Explained:
1. **Document Loading**: Reads 24 PDF manuals page-by-page and attaches hidden metadata (`source` filename and `page` number).
2. **Text Chunking**: Breaks large documents into manageable 3000-character segments with 600-character overlap so sentences aren't split awkwardly.
3. **Embedding**: NVIDIA's embedding model translates text chunks into dense mathematical vectors.
4. **Vector Caching**: Saves vectors to a local `faiss_index/` folder on disk. Future runs start in **< 0.1 seconds** without calling the embedding API again!
5. **Retrieval**: When a query is submitted, FAISS performs vector similarity search to fetch the top 5 most relevant chunks.
6. **Prompt & Generation**: Chunks and the question are injected into an LCEL prompt pipeline and sent to **Google Gemini**.
7. **Structured Parsing & Guardrails**: Enforces a strict Pydantic JSON schema. If the question is outside Love2D (e.g., C#), it cleanly explains the topic is out-of-scope and suppresses irrelevant context.

---

## 🛠️ 3. Tech Stack & Models Used

| Component | Technology / Model | Purpose in Project |
| :--- | :--- | :--- |
| **Orchestration** | **LangChain (LCEL)** | Manages pipelines using clean pipe `\|` operators (`retriever \| prompt \| llm \| parser`). |
| **Large Language Model (LLM)** | **Google Gemini 3.1 Flash-Lite** | Generates fast, accurate, structured answers and runnable Lua code snippets. |
| **Embedding Model** | **NVIDIA `nemotron-3-embed-1b`** | Converts text into high-dimensional semantic vectors for conceptual matching. |
| **Vector Database** | **FAISS (Facebook AI Similarity Search)** | In-memory and disk-cached vector store for fast top-$k$ similarity search. |
| **Text Splitter** | **RecursiveCharacterTextSplitter** | Recursively divides PDF text by paragraphs, lines, and spaces (`chunk_size=3000`, `overlap=600`). |
| **PDF Extraction** | **pypdf** | Reads binary PDF files and extracts text page-by-page. |
| **Schema Validation** | **Pydantic** | Validates structured JSON output (`Transformer` model with `text` and `context` fields). |
| **Web User Interface** | **Streamlit** | Responsive, modern web chat UI with session history and `@st.cache_resource`. |
| **Environment Config** | **python-dotenv** | Securely loads `NVIDIA_API_KEY` and `GEMINI_API_KEY`. |

---

## 🧠 4. Core Algorithms & Design Patterns

### 1. Recursive Character Chunking
- **Why it's used**: Simply cutting text every $N$ characters breaks words and code syntax in half.
- **How it works**: It tries to split text at natural boundaries in order: `\n\n` (paragraphs) $\rightarrow$ `\n` (lines) $\rightarrow$ `" "` (words).
- **Chunk Overlap (600 characters)**: Ensures that if a Lua function spans across two chunks, the beginning and ending context are preserved in both chunks.

### 2. Disk Caching of Vector Index
- **Problem**: Re-reading 24 PDFs and calling NVIDIA's embedding API on every script launch takes 20+ seconds and wastes API credits.
- **Solution**: The vector store saves `index.faiss` and `index.pkl` locally. On subsequent runs, it loads directly from disk in **0.05 seconds** at zero cost.

### 3. LangChain Expression Language (LCEL) & Pipe Operators
The query pipeline is composed cleanly using declarative pipe operators:
```python
rag_chain = (
    RunnableParallel({
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(format_docs),
    })
    | prompt
    | llm
    | parser
)
```
- **`RunnableParallel`**: Runs question pass-through and FAISS document retrieval simultaneously.
- **`RunnableLambda`**: Converts retrieved `Document` chunks into clean text.
- **`| prompt | llm | parser`**: Pipes the formatted prompt directly to Gemini and parses the structured response.

### 4. Anti-Hallucination Guardrails
- If a user asks about an unrelated technology (like C# or Java), the model is instructed to state that the topic is not in the documentation and set `'context': ''`.
- The application automatically detects these negative cues and suppresses irrelevant references.

---

## 📊 5. Evaluation & Accuracy Metrics

The system was evaluated against a 10-query benchmark suite via [`evaluate.py`](file:///d:/Python_Rag_Appplication_Final/evaluate.py) testing retrieval, purity, accuracy, and refusal:

| Metric | Score | Plain English Meaning |
| :--- | :---: | :--- |
| **Query Recall (Hit Rate@5)** | **87.5%** (7/8) | In 7 out of 8 in-scope questions, FAISS successfully retrieved the target manual in the top 5 chunks. |
| **Precision@5** | **65.0%** | **Purity**: On average, 65% of the chunks in the top 5 were direct hits from the target manual with minimal filler. |
| **Answer Accuracy** | **100.0%** (8/8) | **100% of answers** correctly covered all ground-truth technical concepts and code logic. |
| **Guardrail Accuracy** | **100.0%** (2/2) | **Zero hallucinations**: 100% of out-of-scope questions (e.g. C# API) were safely refused. |
| **Average Response Time** | **3.97s** | Sub-0.05s local FAISS search + ~3.5s Gemini response generation. |

> *Detailed question-by-question logs are saved in `evaluation_results.csv` and `evaluation_report.md`.*

---

## 📂 6. Project Directory Structure

```plaintext
Python_Rag_Appplication_Final/
├── documents/                  # 24 Love2D PDF manuals (Collisions, Audio, Camera, etc.)
├── faiss_index/                # Cached FAISS index files (index.faiss, index.pkl)
├── config.py                   # Central settings (API keys, paths, model names, chunk sizes)
├── schemas.py                  # Pydantic schemas (Transformer, Answer) & output parser
├── loader.py                   # PDF ingestion and RecursiveCharacterTextSplitter logic
├── vectorstore.py              # NVIDIA embeddings, FAISS indexing, caching & retriever setup
├── chain.py                    # Prompt templates, Gemini LLM client & LCEL RAG chain
├── capstone.py                 # Terminal interactive CLI application
├── app.py                      # Modern Streamlit Web Chat UI
├── evaluate.py                 # Automated accuracy, precision, recall & latency evaluation
├── evaluation_report.md        # Comprehensive benchmark evaluation report
├── evaluation_results.csv      # CSV export of benchmark results
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 7. How to Run & Live Demo Guide

### Step 1: Clone or Navigate to the Workspace
Ensure your command prompt is inside the project directory:
```bash
cd Python_Rag_Appplication_Final
```

### Step 2: Set Up Python Virtual Environment
Activate your existing virtual environment:
```powershell
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\.venv\Scripts\activate.bat
```

### Step 3: Configure Environment Variables
Ensure your `.env` file contains your API keys:
```env
NVIDIA_API_KEY="your-nvidia-api-key-here"
GEMINI_API_KEY="your-gemini-api-key-here"
```

---

### Option A: Run the Live Streamlit Web UI (Recommended for Demo)
Launch the interactive web application:
```bash
streamlit run app.py
```
- Open your browser at **`http://localhost:8501`**.
- Try asking:
  - *"How do I play sound effects in Love2D?"*
  - *"Can you write a code snippet for entity collisions in Lua?"*
  - *"How to write code in C#?"* *(Tests the out-of-scope guardrail!)*

---

### Option B: Run the Terminal CLI
To test through a command-line interface:
```bash
python capstone.py
```
Type your query and press **Enter**. Type `exit` to quit.

---

### Option C: Run the Accuracy & Metric Evaluation Benchmark
To run the automated 10-query benchmark:
```bash
python evaluate.py
```
This prints the full evaluation report directly to your terminal and updates `evaluation_report.md` and `evaluation_results.csv`.

---

## 🔮 8. Future Scope & Roadmap

To take this enterprise game engine assistant from a capstone project to a production-grade studio tool, here are the primary planned improvements:

### 1. 🧠 Conversational Chat Memory (Multi-Turn Dialogue)
- **What it is in simple English**: Currently, every question is treated independently. With conversational memory, the AI remembers what you were talking about.
- **Why it matters**: A developer can ask natural follow-up questions without repeating themselves:
  > *User: "How do I create a player entity in Lua?"*  
  > *User (Follow-up): "Can you add a jump function to **that** player?"*
- **How to implement**: Using LangChain's `create_history_aware_retriever` with a session conversation buffer.

---

### 2. 🔍 Advanced Search: Hybrid Search + Re-ranking
- **What it is in simple English**: 
  - **Hybrid Search**: Combines keyword search (BM25) for exact Lua function names (`love.graphics.newQuad`, `love.audio.newSource`) with FAISS vector search for conceptual questions.
  - **Re-ranking**: A second-stage model (like an NVIDIA Cross-Encoder) inspects the top 15 candidate chunks and picks the top 3 purest chunks.
- **Why it matters**: Filters out background noise and pushes retrieval Precision from **65.0% to over 90%**.

---

### 3. 📂 Multi-Source Data Ingestion (Raw Code & Git Repositories)
- **What it is in simple English**: Expanding beyond PDF manuals to directly read raw game code files.
- **Why it matters**: In real studios, proprietary game engines consist of thousands of `.lua` scripts, `.md` markdown design docs, and Git repositories. Indexing actual codebase files allows junior developers to search live function definitions across the entire studio repository.

---

### 4. 🖼️ Multimodal RAG (Images, Sprite Sheets & Diagrams)
- **What it is in simple English**: Allowing the AI to understand and search **both text and images** at the same time.
- **Why it's a game-changer for Game Development**:
  - Video game documentation is heavily visual! The manuals contain **sprite sheets**, **collision bounding box diagrams**, **tilemap grids**, and **camera viewport charts**.
  - **Visual Manual Search**: The AI can retrieve and display the actual visual diagrams from the PDFs (e.g., showing the diagram of how coordinate axes `(x, y)` work in 2D space).
  - **Screenshot Error Diagnosis**: A game developer can upload a screenshot of a graphical glitch or a sprite sheet into the Streamlit UI, and the AI (using Gemini Vision) can visually diagnose the problem and cite the exact page in the manual.

---

### 5. ⚡ Real-Time Streaming Responses (`chain.stream`)
- **What it is in simple English**: Instead of waiting ~3.5 seconds for the entire response to finish generating in the background, the answer types out word-by-word in real time.
- **Why it matters**: Reduces perceived latency from 3.5 seconds to **under 0.5 seconds**, giving a fluid, responsive chat experience like ChatGPT.

---

### 📋 Future Scope Roadmap Summary:

| Feature | Category | What It Adds in Plain English | Expected Impact |
| :--- | :--- | :--- | :--- |
| **Conversational Memory** | **Dialogue** | Remembers previous questions for natural follow-ups. | Seamless back-and-forth debugging sessions. |
| **Hybrid Search & Re-ranking** | **Search** | Exact function keyword matching + Cross-encoder filtering. | Pushes retrieval Precision from 65% to 90%+. |
| **Multi-Source Data Ingestion** | **Data** | Reads `.lua` scripts, Markdown, and Git repositories. | Indexes actual game engine codebases, not just PDFs. |
| **Multimodal RAG** | **Vision** | Indexes diagrams and allows developers to upload game screenshots. | Visual debugging of sprite sheets, tiles, and hitboxes. |
| **Response Streaming** | **UI / Speed** | Words stream onto the screen one-by-one in real-time. | Perceived response time drops to < 0.5 seconds. |

