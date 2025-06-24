# rag-pdf-learner
An super duper tiny **interactive learning app** where users can **speak questions aloud** and get **spoken answers** from the contents of a **pdf document**. 

### How it works
- **Speech-to-Text** and **Text-to-Speech**:  
	  - The voice speech synthesis is done using Deepgram and Cartesia, this is a two way process.
	  - First the voice is transcribed into text and then, the speech synthesis is generated one Gemini generated a contextual answer.
  
- **PDF Parsing**:  
    - **PyPDF2** to extract raw text from uploaded PDF documents (irrelevant).
    
- **Embeddings & Retrieval**: 
    - Text is **split into chunks**. 
    - Embeddings generated via **SentenceTransformers** (`all-MiniLM-L6-v2` model). 
    - **Cosine similarity** is computed between spoken queries and chunk embeddings to retrieve top relevant content.
	
- **Language Model for Response**:  
    - Uses **Google Gemini 1.5 Flash API** to generate a concise, contextual answer from the most relevant PDF content.
    
- **Audio Playback**:  
    - **PyDub** is used to play back the TTS response audio seamlessly.

### How to use

1. Install dependencies with pip using the requirements file.
2. Add API keys from .env.example file.
3. Run with:
```bash
    python3 app.py <path-to-pdf> 
```
