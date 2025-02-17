import re

class ResponseFormatter:
    def __init__(self):
        """
        Initializes the response formatter.
        """
        pass

    def clean_text(self, text):
        """
        Cleans the response by removing unwanted characters and extra spaces.
        :param text: Raw LLM-generated response
        :return: Cleaned text
        """
        text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
        text = text.replace("\n", " ")  # Convert newlines to spaces
        return text.strip()

    def format_response(self, response):
        """
        Formats the response into structured sections.
        :param response: Cleaned LLM-generated response
        :return: Structured response
        """
        if ":" in response:
            sections = response.split(":")
            formatted_response = "\n".join([f"🔹 **{sections[i].strip()}**: {sections[i+1].strip()}" 
                                            for i in range(0, len(sections)-1, 2)])
        else:
            formatted_response = f"✅ **Response:** {response}"
        
        return formatted_response

# Example usage
if __name__ == "__main__":
    formatter = ResponseFormatter()
    
    raw_response = "The termination clause states: Either party may terminate this contract with a 30-day written notice."
    cleaned_response = formatter.clean_text(raw_response)
    structured_response = formatter.format_response(cleaned_response)

    print("🔹 Final Formatted Response:\n", structured_response)
