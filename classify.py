import argparse
import os
import datetime
import json
import joblib
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from helper_functions import *
import patoolib
from typing import Union , List
import shutil
 

def unzip_folder(folder_path :str, curr_dir:str):
    """takes an archived file path and unzip it
    :param folder_path: path to the archived file.
    :type folder_path: str
    :param curr_dir: current directory, to put the extracted folder in.
    :type curr_dir: str    
    :returns: path of the new exracted folder 
    :rtype: str
    """
    print("[INFO] Extracting the archived file...")
    patoolib.extract_archive(folder_path, outdir=curr_dir)
    print("[INFO] Extraction completed.")

    return os.path.join(os.path.join(curr_dir , os.path.basename(os.path.normpath(folder_path)).split('.zip')[0]))

def make_dir(dir_names : Union[List[str] , str]):
    """takes a list fo strings or a string and make a directory based on it
    :param dir_name: the name(s) which will be the path to the directory.
    :type dir_name: Union[List[str] , str]
    :returns: a path to the new directory created 
    :rtype: str
    """
    if type(dir_names) == str:
        if dir_names.strip() == "":
            raise ValueError("Please enter a name to the directory")
   
        os.makedirs(dir_names , exist_ok=True)
        return dir_names
  
    elif type(dir_names) == list and len(dir_names) == 0:
        raise ValueError("Please enter list with names")
  
    elif type(dir_names) == list and len(dir_names) == 1:
        os.makedirs(dir_names[0] , exist_ok=True)
        return dir_names[0]
  
    final_dir = os.path.join(dir_names[0] , dir_names[1])
    for name in dir_names[2:]:
        final_dir = os.path.join(final_dir , name)

    os.makedirs(final_dir , exist_ok=True)
    return final_dir

def get_clip(clip_model_type : str = 'ViT-B-32' , pretrained:str = 'openai'):
    """initiates the clip model, initiates the device type, initiates the preprocess
    :param clip_model_type: type of the CLIP model. 
    :type clip_model_type: str
    :param pretrained: pretrained name of the model.
    :type pretrained: str
    :returns: clip model object , preprocess object , device object
    :rtype: Object , Object , str
    """
    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_type,pretrained=pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return clip_model , preprocess , device

def create_models_dict(models_path:str):
    """take the path of the models' folder, load all of them in one dict
    :param models_path: path to the models pickle files path
    :type models_path: str
    :returns: dictionary contains all the models with their names
    :rtype: Dict
    """
    models_dict = {} # Dictionary for all the models objects
    for model_file in tqdm(os.listdir(models_path)):
        if not model_file.endswith('pkl'):
            continue

        model_pkl_path = os.path.join(models_path , model_file)
        model_name = model_file.split('.pkl')[0]
    
        # Loading model object 
        with open(model_pkl_path, 'rb') as model:
            models_dict[model_name] = joblib.load(model)
        model.close()

    return models_dict

def create_mappings_dict(mappings_path:str):
    """take the path of the mappings' folder, load all of them in one dict
    :param mappings_path: path to the models pickle files path
    :type models_path: str
    :returns: dictionary contains all the models with their names
    :rtype: Dict
    """
    mappings_dict = {} # Dictionary for all the models objects
    for mapping_file in tqdm(os.listdir(mappings_path)):
        if not mapping_file.endswith('json'):
            continue

        mapping_file_path = os.path.join(mappings_path , mapping_file)
        model_name = mapping_file.split('.json')[0]

        with open(mapping_file_path, 'rb') as mapping_obj:
            mappings_dict[model_name] = json.load(mapping_obj)
        mapping_obj.close()

    return mappings_dict
    
def loop_images(folder_path: str  , image_tagging_folder : str ,
                models_dict : dict, class_mapping_dict : dict ,  
                clip_model , preprocess ,
                device):
    """Loop through images' folder, classify each image,  put each image in it's directory.
    :param folder_path: path to the images' folder.
    :type folder_path: type of the images' folder path.
    :param image_tagging_folder: path to the image tagging folder.
    :type image_tagging_folder: str
    :param models_dict: Dictionary for the models' dict.
    :type models_dict: Dict.
    :param class_mapping_dict: dictionary for class mappings jsons.
    :type class_mapping_dict: dict
    :param clip_model: CLIP model object. 
    :type clip_model: CLIP
    :param preprocess: preprocess object from open_clip
    :param device: device of the machine ("cpu" or "cuda")
    :type device: str
    :returns: copy all the images in their classification directories
    :rtype: None
    """
    for img_file in tqdm(os.listdir(folder_path)):
        try :
            # Image path 
            img_file_path = os.path.join(folder_path , img_file) 
            # loop through each model and find the classification of the image.
            for model_name in models_dict:
                try :    
                    image_class = classify_image(img_file_path , models_dict[model_name] , 
                                                class_mapping_dict[model_name] , clip_model ,
                                                preprocess , device )
                except Exception as e:
                    print(f"[WARNING] Problem with file {img_file} in classification.")
                    continue
                
                # Get model name and tag from model's dict keys.
                model_type = model_name.split('-tag-')[0].split('model-')[1] # model type, ex: svm or logistic regression 
                tag_name = model_name.split('-tag-')[1] # tag/class name

                # Find the output folder and create it based on model type , tag name 
                if   model_type.strip() == 'ovr-svm':
                    tag_name_out_folder= make_dir([image_tagging_folder, 'ovr-svm',f'{tag_name}-results', image_class.strip()])
                elif model_type.strip() == 'ovr-logistic-regression':
                    tag_name_out_folder= make_dir([image_tagging_folder, 'ovr-logistic-regression', f'{tag_name}-results',image_class.strip()])
                else:
                    print("[ERROR]  Something went wrong with folder names!")
                    print(f"[ERROR] Please check {model_name}")
                    continue           
                # Copy the file from source to destination 
                shutil.copy(img_file_path, tag_name_out_folder)
        
        except Exception as e  :
            print(f"[ERROR] {e} in file {img_file}")
            continue

def main(folder_path:str):
    """main function to be running, calls other function
    :param folder_path: path to the images' folder or archive file.
    :type foldr_path: str
    :returns: call all the functions in order.
    :rtype: None
    """
    curr_dir = os.getcwd() # get the current directory

    #Check if it's an archived dataset 
    if folder_path.endswith('.zip'): 
        folder_path = unzip_folder(folder_path)

    # Clean the directoy (converts every .GIF to .PNG).
    clean_directory(folder_path)

    # Create the output directory name with time stamp.
    timestamp = datetime.datetime.now() 
    image_tagging_folder_name = f'tagging_output_{timestamp.month}_{timestamp.day}_{timestamp.hour}_{timestamp.minute}'
    image_tagging_folder = make_dir(image_tagging_folder_name)
    print(f"> Output folder {image_tagging_folder_name}")

    # Get CLIP model 
    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-B-32',pretrained= 'openai')

    # Get the dictionary for both models objects and mappings for  each model 
    print("[INFO] Loading the models and mappers")
    models_dict = create_models_dict(os.path.join(curr_dir , 'outputs' , 'models'))
    class_mapping_dict = create_mappings_dict(os.path.join(curr_dir , 'outputs' , 'class_mappings'))
    print("[INFO] Model and mappers are loaded")

    # Looping over all the images in the folder
    print('[INFO] Starting looping through images')
    loop_images(folder_path , image_tagging_folder,
                models_dict , class_mapping_dict , 
                clip_model , preprocess , 
                device)
    print('[INFO] Finished')

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True)
    args = parser.parse_args()

    # Run the main program 
    main(args.directory) 
