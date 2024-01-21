#install ultralytics in terminal (!pip install ultralytics)
from ultralytics import YOLO


# Load a model
model = YOLO("yolov8x-cls.pt")  # load a pretained model

# Use the model (Change question mark(?...?))
results = model.train(data=?path/to/dataest?, epochs=?number of epoces?, imgsz=640)  # train the model

#*This model is for Classification*
