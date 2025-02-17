from fastapi import FastAPI
from pydantic import BaseModel
from retrieval.query_handler import QueryHandler
from generation.llms import LLMs
from generation.response_formatter import ResponseFormatter

# Initialize FastAPI
app = FastAPI(title="RAG API for Business Documents")

# Initialize Components
query_handler = QueryHandler()
llm_handler = LLMs(model_name="mistral")  # Runs locally via Ollama
response_formatter = ResponseFormatter()

# Request Schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/query/")
async def query_rag(request: QueryRequest):
    """
    API Endpoint: Retrieves relevant document chunks & generates LLM response.
    """
    try:
        # Retrieve relevant document sections
        context = query_handler.prepare_context(request.query, top_k=request.top_k)

        # Generate LLM response
        raw_answer = llm_handler.generate_response(context)

        # Format the response
        formatted_answer = response_formatter.format_response(raw_answer)

        return {
            "query": request.query,
            "response": formatted_answer
        }
    
    except Exception as e:
        return {"error": str(e)}

# Run API with: uvicorn src.api.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
