from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model: str = "mistral"  

@app.post("/chat/")
async def chat_with_model(request: ChatRequest):
    response = ollama.chat(model=request.model, messages=[{"role": "user", "content": request.message}])
    return {"response": response['message']}

