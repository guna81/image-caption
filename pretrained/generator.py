# Load model directly
from transformers import AutoProcessor, TFBlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def blip(image):
    print(type(image))
    inputs = processor(image, return_tensors="tf", padding=True)
    outputs = model.generate(**inputs)
    return processor.batch_decode(outputs, skip_special_tokens=True)