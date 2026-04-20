# Mutual Fund Intelligence (RAG System)

This project started from a simple problem: finding specific details like exit load, taxation rules, or risk disclosures inside 100+ page Mutual Fund Scheme Information Documents (SIDs).

Generic LLMs often hallucinate these values or miss context across sections. This system avoids that by grounding every response in the actual document using a Retrieval-Augmented Generation (RAG) pipeline.

---

## Live Demo

https://mutual-fund-chat-rag.streamlit.app/

---

## What This Does

- Lets you upload and query mutual fund PDFs  
- Retrieves relevant sections instead of guessing answers  
- Generates responses strictly based on document context  
- Maintains better accuracy for financial queries  

---

## Technical Approach

### Document Processing
- Used `PyPDFLoader` to extract text from PDFs  
- Applied a recursive character splitter  
- Chunk size: 1000  
- Overlap: 200  

The overlap was important to preserve context in financial tables and section headers.

---

### Vector Storage
- FAISS used for local indexing  
- Chosen for speed and simplicity  
- Works well for single-user or small-scale setups  

---

### Retrieval Strategy
Instead of retrieving a single chunk, the system pulls multiple relevant sections and combines them.

This helps in cases where:
- Risk disclosures are in one section  
- Strategy or conditions are in another  

---

### LLM Choice
- Final responses generated using Gemini Pro  
- Switched from Flash model after observing better reasoning for financial queries  

The goal here was accuracy over latency.

---

## Challenges Faced

### 1. Handling Tables in PDFs
Financial documents contain dense and nested tables.

Problem:
- Standard chunking breaks tables into meaningless text  

Fix:
- Added metadata tags to identify document sections  
- Improved retrieval relevance using section-aware chunks  

---

### 2. Avoiding Misinterpretation
Example:
- "Up to 1%" being interpreted as "1%"  

Fix:
- Used a stronger reasoning model  
- Forced responses to stay grounded in retrieved context  

---

## Project Structure

```bash
mutual-fund-intelligence/
├── app.py              # Streamlit chat interface
├── ingest.py           # Builds FAISS index from PDFs
├── docs/               # Store input PDFs here
├── .env                # API keys
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mutual-fund-intelligence.git
cd mutual-fund-intelligence
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

You need a Groq API key to run the application.

### Option 1: Using Streamlit Secrets

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_api_key_here"
```

### Option 2: Using Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

### Getting an API Key

You can generate a free API key from:  
https://console.groq.com/keys
---

## Running the Project

### Step 1: Add PDFs
Place your mutual fund documents inside the `docs/` folder.

### Step 2: Build index

```bash
python ingest.py
```

### Step 3: Start app

```bash
streamlit run app.py
```

---

## When to Use This

- Comparing mutual funds  
- Checking exit load or tax rules  
- Reading long financial documents faster  
- Avoiding hallucinated financial answers  

---

## Limitations

- Works best with structured PDFs (like SIDs)  
- Not optimized for scanned/image-only PDFs  
- No long-term storage (FAISS is local)  

---

## Future Improvements

- Support for multiple PDFs in one session  
- Highlighting source text in answers  
- Persistent vector database  
- Better table parsing  

---

## Disclaimer

This tool is for informational purposes only. Always verify financial decisions with official documents or advisors.
