import sys
import os

from query_handler import QueryHandler
from llms import LLMs

class RAGPipeline:
    def __init__(self, model_name='mistral:7b-instruct'):
        self.query_handler=QueryHandler()
        self.llms = LLMs(model_name=model_name)

    def answer_query(self,query,top_k=3):
        context=self.query_handler.prepare_context(query,top_k)
        response=self.llms.generate_response(context)

        return response
    

if __name__ == 'main':
    rag_pipeline = RAGPipeline(model_name='mistral')
    query = 'What is the product id?'
    response=rag_pipeline.answer_query(query)

    print("Final answer:")
    print(response)
