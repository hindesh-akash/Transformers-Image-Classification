import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

class ObjectDetection:
    COCO_INSTANCE_CATEGORY_NAMES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', ' toilet',
        'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_image(self, image_path):
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        return img

    def transform_image(self, image):
        return self.transform(image)

    def detect_objects(self, image_path):
        # Load and transform the image
        image = self.load_image(image_path)
        # image = image_path
        image_tensor = self.transform_image(image).unsqueeze(0)
        
        # Perform object detection
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # Parse the output and create the JSON structure
        detected_objects = []
        for label, score, box in zip(predictions['labels'], predictions['scores'], predictions['boxes']):
            if score >= self.threshold:
                try:
                    name = self.COCO_INSTANCE_CATEGORY_NAMES[label.item()],
                except:
                    
                    name = label.item().value,
                    
                detected_objects.append({

                    "bb": {
                        "topLeft": {
                            "x": box[0].item(),
                            "y": box[1].item()
                        },
                        "size": {
                            "width": (box[2] - box[0]).item(),
                            "height": (box[3] - box[1]).item()
                        }
                    },
                    "name" : name[0]
                })

        
        if len(detected_objects) == 0:
            return [
                {
                    "bb":{
                        "topLeft":{
                        "x":0,
                        "y":0
                        },
                        "size":{
                        "width":0,
                        "height":0
                        }
                    },
                    "name":"None"
                }
            ]
        return detected_objects

