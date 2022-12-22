from stage3.classify_helper_functions import *
import os 
from ClipModelLoader import ClipModel
from typing import Union

class ModelLoader:
      
    def __init__(self, tag_name: str, model_type:str = 'torch-logistic-regression', models_folder: str = os.path.join('output','models')):
        self.tag_name      = tag_name
        self.model_type    = model_type
        self.models_folder = models_folder

        self.model = self.__generate_model_obj(generate_model_path(self.models_folder,self.model_type,self.tag_name))


    def __generate_model_obj(self, model_path: str):
        """ generates model dict and get the model object from it."""
        model_dict = create_models_dict(models_path=model_path)          
        return model_dict[f"model-{self.model_type}-tag-{self.tag_name}"]
          
    def CalculateTagWeightFromImageData(self, ImageData: Union[bytes,bytearray] , ClipModel: ClipModel):
            
        image_features = clip_image_features(ImageData,ClipModel.clip,ClipModel.preprocess,ClipModel.device)        
        return classify_image_prob(image_features, self.model ,torch_model=True)
          
    def CalculateTagWeightFromImagefile(self, ImageFilePath: str, ClipModel: ClipModel):
        image_features = clip_image_features(ImageFilePath,ClipModel.clip,ClipModel.preprocess,ClipModel.device)        
        return classify_image_prob(image_features, self.model ,torch_model=True)


def CalculateTagWeightFromImageData(Model: ModelLoader, ImageData: Union[bytes,bytearray] ,ClipModel: ClipModel):
    return Model.CalculateTagWeightFromImageData(ImageData, ClipModel)


def CalculateTagWeightFromImagefile(Model: ModelLoader, ImageFilePath: str ,ClipModel: ClipModel):
    return Model.CalculateTagWeightFromImagefile(ImageFilePath, ClipModel)

