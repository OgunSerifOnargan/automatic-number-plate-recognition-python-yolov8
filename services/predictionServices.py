import cv2
from PIL import Image




def predict_licence_number(img, processor, model):
    image_cv2_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the cv2 image array to PIL Image
    image = Image.fromarray(image_cv2_rgb)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text