# 🎬 Video-Audio Multimodal RAG

A fully local **multimodal Retrieval-Augmented Generation (RAG)** system that:
- 🔍 Extracts visual and audio segments from uploaded videos
- 🧠 Describes keyframes with a vision-language LLM (e.g., GPT-4o-mini)
- 📥 Stores joint image-text embeddings using **Bridgetower-Base** in **LanceDB**
- 🎯 Supports user queries via text, image, or both
- 🖥️ Presents an interactive **Streamlit** frontend with video clip playback and LLM responses

---

## 🧠 Key Features

- **Frame + Transcript + Description**: Automatically extract smart frames with audio transcription and semantic descriptions
- **BridgeTower Embeddings**: Use image-only, text-only, or joint embeddings
- **LanceDB Vector Search**: Store multimodal records and retrieve the best match
- **Multimodel RAG Pipeline**: Route the query to the right retrieval logic
- **Streamlit Frontend**: Upload video ➝ get insight + ask questions + see matching clip

---

## 📁 Project Structure

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/tonywang638156/Multimodel-RAG.git
cd Multimodel-RAG
```

### 2. Set up the environment
```bash
git clone https://github.com/tonywang638156/Multimodel-RAG.git
cd Multimodel-RAG
```

### 3. Get an OpenAI API key
Create a .env file in the project root, replace OPENAI_API_KEY with your own api key
```bash
OPENAI_API_KEY=sk-...
```

### 4. Run the app
```bash
streamlit run app.py
```

## 📷 Supported Queries Samples

| Type         | Supported Input              | Behavior                      |
|--------------|------------------------------|-------------------------------|
| Text-only    | `"What is happening?"`       | Embeds text only              |
| Image-only   | Upload screenshot of a given scene            | Uses default prompt for image |
| Text + Image | Upload scene image + ask question  | Uses both for joint embedding |


## 📦 Data Storage

All records are inserted into **LanceDB**, with the following structure:

```json
{
  "image_path": "frames/frame_04_6707ms.jpg",
  "audio_path": "audio/audio_04_6707ms.wav",
  "description": "The man is confidently explaining...",
  "timestamp_ms": 6707.36
}
```


## 💬 LLM Integration

Uses **GPT-4o** (or other vision-enabled OpenAI models) to:

- 🔍 **Elaborate** on keyframes  
- 💡 **Answer** user queries using retrieved image + semantic description

