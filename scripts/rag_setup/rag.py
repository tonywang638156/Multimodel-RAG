from scripts.vectorstores.raw_multimodal_lancedb import RawMultimodalLanceDB
from scripts.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings

from scripts.MLM.openai_lvlm import OpenAIVLM
from scripts.MLM.gemini_lvlm import GeminiVLM

import base64
from PIL import Image
from io import BytesIO


# Updated prompt
prompt_template = (
    #"Here are retrieved image segments and their associated semantic description:\n\n"
    "Here are retrieved image segments and their associated semantic description:\n\n"
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
#def run_custom_rag_pipeline(user_query: str = "", image_path: str = None, num_of_retrieval: int = 1):
def run_custom_rag_pipeline(user_query: str = "", image_path: str = None, num_of_retrieval: int = 1, full_transcript: str = None):
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
    #llm = OpenAIVLM()
    #llm = OpenAIVLM() if num_of_retrieval == 1 else GeminiVLM()
    if num_of_retrieval == 1:
        llm = OpenAIVLM()
        print("🧠 Using OpenAIVLM (GPT-4o) for single-image query.")
    else:
        llm = GeminiVLM()
        print("🌠 Using GeminiVLM (Gemini 1.5 Pro) for top-3 image RAG.")



    # Perform joint similarity search
    #results = vectorstore.similarity_search(query_text=user_query, image_path=image_path, k=1)
    results = vectorstore.similarity_search(query_text=user_query, image_path=image_path, k=num_of_retrieval)

    if not results:
        return {"llm_output": "❌ No relevant result found."}

    # # Get top result
    # top = results[0]
    # description = top["text"]                    # joint embedding source
    # matched_image_path = top["image_path"]       # for display + GPT-4o vision
    # img_inline = inline_image_as_base64(matched_image_path) #!!!!!!!!!!!!
    # metadata = top["metadata"]                   # all metadata (audio, timestamp...)

    # # Build LLM prompt
    # prompt = (
    #     "Here is the retrieved image:\n"
    #     f"{img_inline}\n\n"
    #     f"📝 Description: '{description}'\n\n"
    #     f"{user_query}"
    # )


    # print("🧾 Prompt sent to OpenAI:\n", prompt)
    # llm_output = llm.invoke({"prompt": prompt, "image": matched_image_path})

    prompt_parts = ["Here are the retrieved image segments and their associated semantic descriptions:\n"]
    vision_images = []

    for idx, res in enumerate(results):
        desc = res["text"]
        img_path = res["image_path"]
        metadata = res["metadata"]
        img_inline = inline_image_as_base64(img_path)
        
        prompt_parts.append(
            f"🔹 **Frame {idx+1}:**\n{img_inline}\n\n📝 Description: '{desc}'\n"
        )
        vision_images.append(img_path)

    prompt_parts.append(f"\n{user_query}")
    if full_transcript and num_of_retrieval > 1:
        prompt_parts.append(f"📄 Full Transcript:\n{full_transcript}\n")
    full_prompt = "\n\n".join(prompt_parts)

    if num_of_retrieval == 1:
        print("🧾 Prompt sent to OpenAI-4o:\n", full_prompt)
    else:
        print("🧾 Prompt sent to Gemini-1.5pro:\n", full_prompt)

    #print("🧾 Prompt sent to OpenAI:\n", full_prompt)

    # 🔁 If your LLM supports multiple images (e.g. GPT-4o multi-image)
    llm_output = llm.invoke({
        "prompt": full_prompt,
        "image": vision_images if len(vision_images) > 1 else vision_images[0]
    })

    # return {
    #     "prompt": prompt,
    #     "image_used": matched_image_path,
    #     "description": description,
    #     "llm_output": llm_output,
    #     "metadata": metadata
    # }
    return {
        "prompt": full_prompt,
        "image_used": vision_images if len(vision_images) > 1 else vision_images[0],
        "description": [res["text"] for res in results],
        "llm_output": llm_output,
        "metadata": [res["metadata"] for res in results]
    }
