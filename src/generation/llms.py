import ollama

class LLMs:
    def __init__(self, model_name = 'mistral'):
        self.model_name = model_name

    def generate_response(self,prompt):
        response = ollama.chat(
            model = self.model_name,
            messages = [{"role":'system',"content":"you are an AI assistant sepcialized in document analysis"},
                        {"role":'user',"content":prompt}]
        )
        return response['message']['content']
    

if __name__ == 'main':
    llms = LLMs(model_name = 'mistral')

    query_context = """Context:
    this file contains invoice of information.
    
    Question: what is the unit price?
    
    Answer:"""

    answer = llms.generate_response(query_context)
    print("generated answer :/n",answer)