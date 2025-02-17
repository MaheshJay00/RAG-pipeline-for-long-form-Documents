import os
import torch 
import json
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class OCRProcessor:
    def __init__(self, detection_model = 'db_resnet50',recognition_model="crnn_vgg16_bn"):

        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

        self.model = ocr_predictor(
            det_arch = detection_model,
            reco_arch = recognition_model,
            pretrained = True
        ).to(self.device)

    def extract_text(self,file_path):


        print(f"processing file:{file_path}")

        if file_path.lower().endswith('.pdf'):
            doc = DocumentFile.from_pdf(file_path)
        else:
            doc = DocumentFile.from_images(file_path)

        result = self.model(doc)
        extracted = result.export()
        return extracted 
    
    def save_extracted(self, extracted,output):

        with open(output,'w',encoding = 'utf-8') as f:
            json.dump(extracted,f,indent=4)

        print(f"extracted is saved to:{output}")


if __name__ == '__main__':
    ocr = OCRProcessor()

    file_path = "Data/CompanyDocuments/invoices/invoice_10248.pdf"
    extracted_text = ocr.extract_text(file_path)

    ocr.save_extracted(extracted_text,"output/extracted_text.json")