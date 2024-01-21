#install in terminal (pip install requirments.txt)
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import numpy as np

# Load the YOLO model
model = YOLO('your/model/best.pt/location')  # Load your custom model

# Load the image and convert it to a PIL Image
image_path = 'your/testing's/image/location'
image = Image.open(image_path)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Specify the expected height and width
    transforms.ToTensor(),
])

# Apply the transformation
input_image = transform(image)

# Predict with the model
results = model.predict(input_image)

#print(results)
names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(f"Your tire is {names_dict[np.argmax(probs)]}")
