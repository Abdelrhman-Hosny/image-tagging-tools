from stage3.classify_helper_functions import *
from stage2.train_helper_functions import *
from model_ovr_logistic_regression_classifier import OvrLogisticRegressionModel
from model_ovr_svm_classifier import OvrSvmModel
from model_pytorch_logistic_regression_classifier import PytorchLogisticRegressionModel 
import os 


class ModelLoader:

    def __init__(self, pipeline_data_directory: str):
        self.pipeline_data_directory = pipeline_data_directory
        self.models_folder = os.path.join(self.pipeline_data_directory, "models")

    def listModels(self):
        """ list all the model we have in outputs/model """
        for sub_dir in os.listdir(self.models_folder):
            
            # check if it's a .pkl file or sub directory
            if os.path.isfile(os.path.join(self.models_folder, sub_dir)) and sub_dir.endswith('.pkl'):
                model_type, tag_name = get_model_tag_name(os.path.splitext(sub_dir)[0])
                print(f"model-type = {model_type}, tag= {tag_name}")
                continue
            
            for model in os.path.join(self.models_folder, sub_dir):
                model_type, tag_name = get_model_tag_name(os.path.splitext(model)[0])
                print(f"model-type = {model_type}, tag= {tag_name}")
    
    def listModelTypes(self):       
        for sub_dir in os.listdir(self.models_folder):
            if os.path.isdir(os.path.join(self.models_folder,sub_dir)):
                print(f"model-type {sub_dir}")
    
    
    def listTags(self, model_type: str):
        model_type_folder = os.path.join(self.models_folder, model_type)

        for model in os.listdir(model_type_folder):
            _ , tag_name = get_model_tag_name(os.path.splitext(model)[0])
            print(f"tag= {tag_name}")



    def LoadModel(self, tag_name: str , model_type: str):
        """Load a model based on tag name and model type"""
        
        # check whether we have sub directory or not yet.
        if os.path.isdir(os.path.join(self.models_folder, model_type)):
            model_type_folder = os.path.join(self.models_folder, model_type)
        else:
            model_type_folder = self.models_folder

        if "ovr-svm" in model_type:
            return OvrSvmModel(tag_name=tag_name, model_type=model_type, models_folder= model_type_folder)
        elif "ovr-logistic-regression" in model_type:
            return OvrLogisticRegressionModel(tag_name=tag_name, model_type=model_type, models_folder= model_type_folder)
        elif "torch-logistic-regression" in model_type:
            return PytorchLogisticRegressionModel(tag_name=tag_name, model_type=model_type, models_folder= model_type_folder)
        else:
            raise ValueError("[ERROR] Invalid model type.") 
        

