# WanderWise RAG Assistant 

WanderWise is a travel-focused RAG assistant that helps users ask questions from local travel documents.

Users can upload city guides, travel notes, itineraries, or PDF articles, then ask questions about attractions, budgets, transportation, and trip planning.

## Features

- Upload `.pdf`, `.txt`, and `.md` files
- Extract text from uploaded documents
- Split documents into overlapping chunks
- Generate embeddings using `all-MiniLM-L6-v2`
- Store embeddings in a FAISS vector index
- Retrieve the most relevant chunks for a question
- Generate answers using OpenAI
- Display sources with chunk index and similarity score

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- OpenAI API
- PyPDF
- RAG

##  How to Run

1. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your OpenAI API key**
   Create a file named `.env` in the project folder and add:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Upload your travel documents (`.pdf`, `.txt`, `.md`) or use the examples in `sample_docs/`, then start asking questions 

