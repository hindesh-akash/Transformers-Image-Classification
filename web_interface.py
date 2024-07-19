import gradio as gr
from object_detection2 import ObjectDetection
from image_captioning import ImageCaptioning

object_detector = ObjectDetection()
image_captioner = ImageCaptioning()

def process_image(image):
    objects = object_detector.detect_objects(image)
    description = image_captioner.generate_caption(image)
    return objects, description

interface = gr.Interface(
    fn=process_image,
    inputs="image",
    outputs=["json", "text"],
    title="Image Classification Service",
    description="Upload an image to detect objects and generate a scene description."
)

if __name__ == "__main__":
    interface.launch()
