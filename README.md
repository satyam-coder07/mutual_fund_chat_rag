# 🏦 Mutual Fund Intelligence (RAG System)

I built this RAG (Retrieval-Augmented Generation) application to solve a personal frustration: trying to find specific exit load or tax implications in 100+ page Mutual Fund Scheme Information Documents (SID). 

Most generic LLMs hallucinate these numbers; this system ensures every answer is grounded in the actual PDF.

### 🏗️ My Technical Approach
* **Data Parsing:** I used `PyPDF` with a custom recursive character splitter. I found that financial documents have dense tables, so I tuned the chunk size to 1000 characters with a 200-character overlap to keep table headers in context.
* **Vector Storage:** Used **FAISS** for local indexing. It’s lightweight and faster than a cloud DB for a personal research tool.
* **The Retrieval Logic:** I implemented a "Refine" chain. Instead of just grabbing one chunk, the system looks at multiple sections to ensure that a 'Risk' disclosure from page 10 is balanced with a 'Strategy' note from page 50.

### 🛠️ Key Challenges I Fixed
* **Handling Nested Tables:** Standard RAG often breaks tables into meaningless strings. I improved retrieval by adding metadata tags to chunks that identified which section of the SID they belonged to.
* **Accuracy over Speed:** I switched the final generation from Flash to **Gemini Pro** because financial queries require deeper reasoning to avoid misinterpreting "Up to 1%" as "Exactly 1%".

### 🚦 Quick Start
1. Place your Fund PDFs in the `/docs` folder.
2. Run `python ingest.py` to build the vector index.
3. Launch the chat: `streamlit run app.py`
