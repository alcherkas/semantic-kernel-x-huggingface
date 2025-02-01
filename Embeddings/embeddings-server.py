from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import base64
from typing import List

app = FastAPI()

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = "nomic-ai/modernbert-embed-base"

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
model = AutoModel.from_pretrained("nomic-ai/modernbert-embed-base")

@app.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    try:
        texts = [request.input] if isinstance(request.input, str) else request.input
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            output = model(**encoded)
        
        # Mean pooling
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        embeddings = torch.sum(output.last_hidden_state * attention_mask, 1) / attention_mask.sum(1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to base64
        return {
            "data": [
                {
                    "embedding": base64.b64encode(e.numpy().astype(np.float32).tobytes()).decode(),
                    "index": i
                }
                for i, e in enumerate(embeddings)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)