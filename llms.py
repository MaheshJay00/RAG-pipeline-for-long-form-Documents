import ollama
import requests

class LLMs:
    def __init__(self, model_name = 'mistral:7b-instruct'):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate" 

    def generate_response(self,prompt):

        payload={
            "model":self.model_name,
            "prompt":prompt,
            "stream":False
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response received.")
        except requests.exceptions.RequestException as e:
            return f"Error communicating with LLM: {e}"

        
    

if __name__ == 'main':
    llms = LLMs()
    test_prompt = "What is the termination clause in a contract?"
    response = llms.generate_response(test_prompt)
    print("generated answer :/n",response)