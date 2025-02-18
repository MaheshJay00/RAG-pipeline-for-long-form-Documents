from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag import RAGPipeline

app = FastAPI(title = "RAG powered document QA")
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return{"messsage:""welcome to the RAG API"}

@app.get("/query/")
def query_rag(query: str = Query(...,desciption = "the user's question"), top_k: int=5):

    response = rag_pipeline.answer_query(query,top_k)
    return{"query":query,"response":response}

@app.post("/query/")
def query_rag_post(request: QueryRequest):
    return {"query": request.query, "response": rag_pipeline.answer_query(request.query, request.top_k)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
