import json

class TextPreprocessor:
    def __init__ (self):
        pass
    def preprocess_text(self, extracted):
        text_output = []
        for page in extracted.get('pages',[]):
            for block in page.get('blocks',[]):
                for line in block.get('lines',[]):
                    line_text = " ".join([word['value'] for word in line.get("words",[])])
                    text_output.append(line_text)

        return "\n".join(text_output)
    
    def save_preprocessed_text(self,structured_text,output):
        with open(output,'w',encoding = 'utf-8') as f:
            f.write(structured_text)
        print(f"preprocessed saved to:{output}")

if __name__ == "__main__":
    text_processor = TextPreprocessor()

    extracted_text_file = "output/extracted_text.json"
    with open(extracted_text_file,'r',encoding = 'utf-8') as f:
        extracted = json.load(f)

    structured_text = text_processor.preprocess_text(extracted)

    text_processor.save_preprocessed_text(structured_text,'output/structured_text.txt')