from model_loader import ModelLoader
from ClipTools import ClipModel
import os 
import warnings
warnings.filterwarnings('ignore')

# Initializing model loader.
model_loader_obj = ModelLoader("output")
model_loader_obj.listModels()

# Generating classification model for tag ="pos-pixel-art" and model type = "ovr-svm".
classification_model = model_loader_obj.LoadModel("pos-pixel-art", "ovr-svm")

# Loading CLIP model.
clip_model = ClipModel() 

# getting tag weight for an image file path.
tag_weight = classification_model.CalculateTagWeightFromImagefile(os.path.join("test_images", 'example1.jpg'), clip_model)

print(f"tag weight = {tag_weight}")

