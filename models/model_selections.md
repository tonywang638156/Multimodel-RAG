Multimodal RAG System - Model Summary
=====================================

1. 🔍 Embedding Model
----------------------
**Model:** BridgeTower  
**Type:** Vision-Language Multimodal Embedding  
**Purpose:**  
  - Generate joint image-text embeddings for video frames and descriptions.
  - Supports multimodal similarity search using both image and text input.

2. 🧠 Language Models
----------------------

A. GPT-4o (via OpenAI API)  
   - **Usage Case:** When only the top-1 most relevant image-description pair is retrieved.  
   - **Capabilities:**  
     - Handles vision + text inputs.  
     - Generates answers grounded in the retrieved frame and description.  
     - Supports a single image input with a prompt.

B. Gemini 1.5 Pro (via Google Generative AI API)  
   - **Usage Case:** When top-3 image-description pairs are retrieved.  
   - **Capabilities:**  
     - Supports multi-image input.
     - Accepts long prompts and context (e.g. entire video transcript).
     - Ideal for deeper reasoning across multiple visual-textual sources.

3. 🗣️ Speech-to-Text
----------------------
**Model:** OpenAI Whisper (`whisper-1`)  
**Purpose:**  
  - Transcribes audio from the entire uploaded video.  
  - Also used to extract audio segments for per-frame descriptions.  
  - Transcripts are integrated as context during LLM prompting.

4. 🧠 Frame-Level Description Generator
----------------------------------------
**Model:** GPT-4o-mini  
**Input:**  
  - A single image frame  
  - Its local audio transcript  
**Output:**  
  - Semantic visual description used to populate the LanceDB vector store.

5. 🧱 Vector Store
------------------
**Backend:** LanceDB  
**Stored Items:**  
  - Embeddings of image-description pairs  
  - Metadata (timestamp, transcript, etc.)  
**Search Type:** Cosine similarity over multimodal embeddings.

