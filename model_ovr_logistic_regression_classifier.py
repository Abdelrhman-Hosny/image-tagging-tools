import sys
sys.path.insert(0, './')
from stage3.classify_helper_functions import *
from stage2.train_helper_functions import *
import os 
from ClipTools import ClipModel
from typing import Union
import numpy as np

class OvrLogisticRegressionModel:
      
    def __init__(self, tag_name: str, model_type:str = 'ovr-logistic-regression', models_folder: str = os.path.join('output','models')):
        self.tag_name      = tag_name
        self.model_type    = model_type
        self.models_folder = models_folder
        self.is_torch = "torch" in model_type
        self.model = self.__generate_model_obj(generate_model_path(self.models_folder,self.model_type,self.tag_name))

    def __generate_model_obj(self, model_path: str):
        """ generates model dict and get the model object from it."""
        model_dict = create_models_dict(models_path=model_path)          
        return model_dict[f"model-{self.model_type}-tag-{self.tag_name}"]
          
    def CalculateTagWeightFromImageData(self, ImageData: Union[bytes,bytearray] , ClipModel: ClipModel):
        image_features = clip_image_features(ImageData,ClipModel.clip,ClipModel.preprocess,ClipModel.device)        
        return classify_image_prob(image_features, self.model ,torch_model=self.is_torch)

    def CalculateTagWeightFromImagefile(self, ImageFilePath: str, ClipModel: ClipModel):
        image_features = clip_image_features(ImageFilePath,ClipModel.clip,ClipModel.preprocess,ClipModel.device)        
        return classify_image_prob(image_features, self.model ,torch_model=self.is_torch)

    def CalculateTagWeightFromClipFeatures(self, ImageFeatures: np.ndarray):
        return classify_image_prob(ImageFeatures, self.model ,torch_model=self.is_torch)


def CalculateTagWeightFromImageData(Model: OvrLogisticRegressionModel, ImageData: Union[bytes,bytearray] ,ClipModel: ClipModel):
    return Model.CalculateTagWeightFromImageData(ImageData, ClipModel)


def CalculateTagWeightFromImagefile(Model: OvrLogisticRegressionModel, ImageFilePath: str ,ClipModel: ClipModel):
    return Model.CalculateTagWeightFromImagefile(ImageFilePath, ClipModel)


def CalculateTagWeightFromClipFeatures(Model: OvrLogisticRegressionModel ,ImageFeatures: np.ndarray):
    return  Model.CalculateTagWeightFromClipFeatures(ImageFeatures)