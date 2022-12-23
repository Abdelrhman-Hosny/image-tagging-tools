from stage3.classify_helper_functions import * 
import open_clip
from typing import Union
import numpy as np

class ClipModel:
    ''' ClipModel class to get all clip model , preprocess and device '''
    def __init__(self, clip_model: str = 'ViT-B-32', pretrained:str = 'openai'):
        
        self.clip_model = clip_model
        self.pretrained = pretrained
        
        self.clip , self.preprocess , self.device = get_clip(self.clip_model, self.pretrained)

    def download_model(self, model_name: str, pretrained: str):
        """ dowload specifc clip model to the machine. """
        if model_name is None or pretrained is None:
            raise ValueError("[ERROR] please enter the model type.")
        
        open_clip.create_model(model_name = 'ViT-B-32', pretrained= 'openai')
        print("[INFO] Model downloaded succesfully")
    

    def encode_image_from_image_file(self,image_file_path: str):
        """ encodes image with CLIP and returns ndArray of image features. """
        return clip_image_features(image_file_path, self.clip ,self.preprocess,self.device)

    def encode_image_from_image_data(self, image_data: Union[bytes,bytearray] ):
        """ enconding image data with CLIP and returns ndArray of image features """
        return clip_image_features(image_data,self.clip ,self.preprocess,self.device)
    
    def encode_image_list(self,image_list: Union[List[str], List[bytes], List[bytearray]]):
        """encoding a list of images with CLIP and returns a ndArray of all of their embeddings"""
        return np.stack((clip_image_features(image,self.clip,self.preprocess,self.device) for image in image_list), axis=0)








