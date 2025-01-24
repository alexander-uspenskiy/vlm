import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import warnings

warnings.filterwarnings("ignore", message="Some kwargs in processor config are unused")

def upload_and_describe_image(image_path):
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    
    image = Image.open(image_path)
    
    prompt = "Describe the content of this <image> in detail, give only answers in a form of text"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )
    
    description = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return description.strip()

if __name__ == "__main__":
    image_path = "images/bender.jpg"
    
    try:
        description = upload_and_describe_image(image_path)
        print("Image Description:", description)
    except Exception as e:
        print(f"An error occurred: {e}")