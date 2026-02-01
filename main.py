from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (adjust later for production domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check 
@app.get("/")
def health():
    return {"status": "ok"}

# Request body schema
class QueryRequest(BaseModel):
    question: str

# Chat endpoint (lazy RAG load)
@app.post("/chat")
async def chat(request: QueryRequest):
    from retrieval import rag_pipeline 
    answer = rag_pipeline(request.question)
    return {"answer": answer}
