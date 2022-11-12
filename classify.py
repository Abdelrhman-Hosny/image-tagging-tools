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


def main(folder_path):
    
    CURR_DIR = os.getcwd() # get the current directory 
    MODEL_PATH = os.path.join(CURR_DIR , 'outputs' , 'models')  # the path for all the models 
    MAPPING_PATH = os.path.join(CURR_DIR , 'outputs' , 'class_mappings')

    if folder_path.endswith('.zip'): # It's a zip file 
        if os.name == 'nt':
            print("Please unzip the folder first")
            return 
        os.system(f'unzip  {folder_path} -d {CURR_DIR}')
        folder_path = os.path.join(CURR_DIR , os.path.basename(os.path.normpath(folder_path)).split('.zip')[0])
    
    clean_directory(folder_path)
    timestamp = datetime.datetime.now() # timrstampe for the folder 
    # create output folder with time stamp 
    image_tagging_folder_name = f'tagging_output_{timestamp.month}_{timestamp.day}_{timestamp.hour}_{timestamp.minute}'
    image_tagging_folder = os.path.join(CURR_DIR ,image_tagging_folder_name )
    os.makedirs(image_tagging_folder , exist_ok=True) # Make the output folder if it's not here
    print(f"> Output folder {image_tagging_folder_name}")

    # Create open clip model 
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading the models and mappers")
    # dictionary loaded with each model 
    models_dict = {}
    # class mapping dict
    class_mapping_dict = {}
    # Loading all the models and mappings 
    for model_file in tqdm(os.listdir(MODEL_PATH)):
        if not model_file.endswith('pkl'):
            continue

        model_pkl_path = os.path.join(MODEL_PATH , model_file)
        model_name = model_file.split('.pkl')[0]
        
        # Loading model object 
        with open(model_pkl_path, 'rb') as model:
            models_dict[model_name] = joblib.load(model)
        model.close()

        # Loading model mapping
        mapping_file = os.path.join(MAPPING_PATH , f'{model_name}.json')
        with open(mapping_file, 'rb') as mapping_file_obj:
            class_mapping_dict[model_name] = json.load(mapping_file_obj) 
        mapping_file_obj.close()
    print("[INFO] Model and mappers are loaded")

    # create svm and LR folder in image tagging folder 
    svm_folder = os.path.join(image_tagging_folder , 'ovr-svm')
    lr_folder = os.path.join(image_tagging_folder , 'ovr-logistic-regression')
    os.makedirs(svm_folder , exist_ok=True)
    os.makedirs(lr_folder , exist_ok=True)
    
    print('[INFO] Starting looping through images')

    # Looping over all the images in the folder
    for img_file in tqdm(os.listdir(folder_path)):
        try :
            img_file_path = os.path.join(folder_path , img_file) 
            for model_name in models_dict:
                try :    
                    image_class = classify_image(img_file_path , models_dict[model_name] , 
                                            class_mapping_dict[model_name] , clip_model ,
                                            preprocess , device )
                except Exception as e:
                    print(f"[WARNING] Problem with file {img_file}")
                    continue
                model_type = model_name.split('-tag-')[0].split('model-')[1]
                tag_name = model_name.split('-tag-')[1]

                if model_type.strip() == 'ovr-svm':
                    tag_name_out_folder = os.path.join(svm_folder,f'{tag_name}-results')
                    os.makedirs(tag_name_out_folder  , exist_ok=True)
                elif model_type.strip() == 'ovr-logistic-regression':
                    tag_name_out_folder = os.path.join(lr_folder,f'{tag_name}-results')
                    os.makedirs(tag_name_out_folder  , exist_ok=True)
                else:
                    print("something went wrong with folder names !")
                    print(f"please check {model_name}")
                    continue
                
                os.makedirs(tag_name_out_folder , exist_ok = True) # creating the tag results folder 
                # Creating the two folders for tag and other 
                os.makedirs(os.path.join(tag_name_out_folder ,class_mapping_dict[model_name]['0'].strip()) ,exist_ok=True)
                os.makedirs(os.path.join(tag_name_out_folder ,class_mapping_dict[model_name]['1'].strip()) ,exist_ok=True)
                out_folder = os.path.join(tag_name_out_folder , image_class.strip())
                # move the image to the output folder of it now
                if os.name == 'nt': # if it is windows server 
                    os.system(f'copy  {img_file_path} {out_folder}')
                else: # Linux server 
                    os.system(f'cp -r {img_file_path} {out_folder}')
        except Exception as e  :
            print(f"[ERROR] {e} in file {img_file}")
            continue
    print('[INFO] Finished')

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--directory', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()
    # Run the main program 
    main(args.directory) 
