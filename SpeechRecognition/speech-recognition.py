import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
from pathlib import Path
import tempfile
import os

app = FastAPI(title="OpenAI-Compatible ASR API")

whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device= -1)

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio from the uploaded file."""
    try:
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, file.filename)

        # Write the audio file content
        with open(input_path, "wb") as f:
            f.write(await file.read())

        transcription = whisper_model(input_path)["text"]
        return {"text": transcription}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
