@app.get("/v1/bot/embeddings/search")
async def search_text(query: str):
    try:
        embeddings = get_embeddings(query)
        results = search_embeddings(embeddings)
        if results:
            return {"results": results}
        else:
            return {"message": "No results found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# test

import faiss
import numpy as np
import pandas as pd

index_file = "faiss_index.bin"
data_file = "data.csv"


# Load or initialize the FAISS index
def load_faiss_index():
    try:
        index = faiss.read_index(index_file)
        data = pd.read_csv(data_file)
    except Exception as e:
        print(f"Failed to load FAISS index or data file: {e}")
        index = faiss.IndexFlatL2(1536)
        data = pd.DataFrame(columns=["text", "embedding"])
    return index, data


def save_faiss_index(index, data):
    faiss.write_index(index, index_file)
    data.to_csv(data_file, index=False)


index, data = load_faiss_index()


def save_to_faiss(text: str, embeddings):
    global index, data
    embedding_np = np.array(embeddings).astype("float32")
    index.add(np.array([embedding_np]))
    new_data = pd.DataFrame({"text": [text], "embedding": [embedding_np.tolist()]})
    data = pd.concat([data, new_data], ignore_index=True)
    save_faiss_index(index, data)


def search_in_faiss(embeddings):
    global index, data
    embedding_np = np.array(embeddings).astype("float32")
    D, I = index.search(np.array([embedding_np]), 3)
    results = [data.iloc[i]["text"] for i in I[0] if i != -1]
    return results
