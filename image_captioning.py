from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

class ImageCaptioning:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def generate_caption(self, image_path):
        image = Image.fromarray(image_path.astype('uint8'))
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        output_ids = self.model.generate(pixel_values)
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
