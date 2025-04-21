from vectorstores.raw_multimodal_lancedb import RawMultimodalLanceDB
from embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from MLM.openai_lvlm import OpenAIVLM
import base64
from PIL import Image
from io import BytesIO


# Updated prompt
prompt_template = (
    "Here is a retrieved image segment and its associated semantic description:\n\n"
    "📝 Description: '{description}'\n\n"
    "{user_query}"
)

def inline_image_as_base64(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((600, 400))  # or smaller to reduce token size
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=30)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"![frame](data:image/jpeg;base64,{b64})"


#def run_custom_rag_pipeline(user_query: str, image_path: str = None):
def run_custom_rag_pipeline(user_query: str = "", image_path: str = None):
    # 👇 Default text for image-only queries
    if not user_query and image_path:
        print("##############image-only query works!!!!!!!!!!!!!!!")
        user_query = "Describe the contents of this image and trace it back to its video." #(test-by-tony)
    embedder = BridgeTowerEmbeddings()
    vectorstore = RawMultimodalLanceDB(
        uri="./shared_data/.lancedb",
        embedding_model=embedder,
        table_name="demo_tbl"
    )
    llm = OpenAIVLM()

    # Perform joint similarity search
    results = vectorstore.similarity_search(query_text=user_query, image_path=image_path, k=1)

    if not results:
        return {"llm_output": "❌ No relevant result found."}

    # Get top result
    top = results[0]
    description = top["text"]                    # joint embedding source
    matched_image_path = top["image_path"]       # for display + GPT-4o vision
    img_inline = inline_image_as_base64(matched_image_path) #!!!!!!!!!!!!
    metadata = top["metadata"]                   # all metadata (audio, timestamp...)

    # Build LLM prompt
    prompt = (
        "Here is the retrieved image:\n"
        f"{img_inline}\n\n"
        f"📝 Description: '{description}'\n\n"
        f"{user_query}"
    )
    # prompt = prompt_template.format(
    #     description=description,
    #     user_query=user_query
    # )

    print("🧾 Prompt sent to OpenAI:\n", prompt)
    llm_output = llm.invoke({"prompt": prompt, "image": matched_image_path})

    return {
        "prompt": prompt,
        "image_used": matched_image_path,
        "description": description,
        "llm_output": llm_output,
        "metadata": metadata
    }