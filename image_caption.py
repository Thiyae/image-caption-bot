from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import argparse

def generate_caption(image_path):
    """
    Generate a caption for the given image using BLIP model.
    
    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        str: Generated caption or error message.
    """
    try:
        # Load BLIP model and processor
        model_id = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        
        # Set device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Generate caption
        outputs = model.generate(**inputs, max_length=16, num_beams=4)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a caption for an image using BLIP model.")
    parser.add_argument("image_path", type=str, help="Path to the input image file (e.g., image.jpg)")
    args = parser.parse_args()
    
    # Generate and print caption
    caption = generate_caption(args.image_path)
    print(f"Caption: {caption}")

if __name__ == "__main__":
    main()