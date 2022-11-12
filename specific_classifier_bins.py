import argparse
import os
import datetime
import json
import joblib
import open_clip
import torch
from tqdm import tqdm
from helper_functions import *


def main(folder_path , model_file_path):
    """
    Runs the whole process 

    :folder_path: path to images' directory or images' zip file 
    :model_file_path: path to model's pickle file 
    """

    CURR_DIR = os.getcwd() # get the current directory 
    MAPPING_PATH = os.path.join(CURR_DIR , 'outputs' , 'class_mappings')

    if folder_path.endswith('.zip'): # It's a zip file 
        if os.name == 'nt':
            print("Please unzip the folder first")
            return 
        os.system(f'unzip  {folder_path} -d {CURR_DIR}')
        folder_path = os.path.join(CURR_DIR , os.path.basename(os.path.normpath(folder_path)).split('.zip')[0])
    
    clean_directory(folder_path) # Clean the directory from GIFs to PNG 

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
    
    if not model_file_path.endswith('pkl'):
        print("[ERROR] This is not a pickle file")
        return 
    
    model_name = os.path.basename(model_file_path).split('.pkl')[0]
    # Loading model object 
    with open(model_file_path, 'rb') as model:
        models_dict[model_name] = joblib.load(model)
    model.close()

    # Loading model mapping
    mapping_file = os.path.join(MAPPING_PATH , f'{model_name}.json')
    with open(mapping_file, 'rb') as mapping_file_obj:
        class_mapping_dict[model_name] = json.load(mapping_file_obj) 
    mapping_file_obj.close()
    print("[INFO] Model and mappers are loaded")

    model_type = model_name.split('-tag-')[0].split('model-')[1]
    # create svm and LR folder in image tagging folder
    if model_type.strip() ==  'ovr-svm' : 
        model_type_folder = os.path.join(image_tagging_folder , 'ovr-svm')
        os.makedirs(model_type_folder , exist_ok=True)
    elif model_type.strip() == 'ovr-logistic-regression':
        model_type_folder = os.path.join(image_tagging_folder , 'ovr-logistic-regression')
        os.makedirs(model_type_folder , exist_ok=True)
    
    print('[INFO] Starting looping through images')

    # Looping over all the images in the folder
    for img_file in tqdm(os.listdir(folder_path)):
        try :
            img_file_path = os.path.join(folder_path , img_file) 
            try :    
                class_bin_dict = classify_image_bin(img_file_path , models_dict[model_name] , 
                                        class_mapping_dict[model_name] , clip_model ,
                                        preprocess , device )
            except Exception as e:
                print(f"[WARNING] Problem with file {img_file}")
                continue
            #model_type = model_name.split('-tag-')[0].split('model-')[1]
            tag_name = model_name.split('-tag-')[1]

            tag_name_out_folder = os.path.join(model_type_folder,f'{tag_name}-results')
            os.makedirs(tag_name_out_folder  , exist_ok=True)
            
            os.makedirs(tag_name_out_folder , exist_ok = True) # creating the tag results folder 
            # Creating the two folders for tag and other 
            os.makedirs(os.path.join(tag_name_out_folder ,class_mapping_dict[model_name]['0'].strip()) ,exist_ok=True)
            os.makedirs(os.path.join(tag_name_out_folder ,class_mapping_dict[model_name]['1'].strip()) ,exist_ok=True)
            ## we have for example svm-->tag-results-->tag , other
            classes = list(class_bin_dict.keys())
            first_bin_out_folder  = os.path.join(tag_name_out_folder  , classes[0] , class_bin_dict[classes[0]]) # out for the first class bin
            second_bin_out_folder = os.path.join(tag_name_out_folder  , classes[1] , class_bin_dict[classes[1]]) # out for the second class bin
            
            os.makedirs(first_bin_out_folder  , exist_ok=True) # create first bin folder 
            os.makedirs(second_bin_out_folder , exist_ok=True) # Create second bin folder 

            # move the image to the output folder of it now
            if os.name == 'nt': # if it is windows server 
                os.system(f'copy  {img_file_path} {first_bin_out_folder}')
                os.system(f'copy  {img_file_path} {second_bin_out_folder}')
            else: # Linux server 
                os.system(f'cp -r {img_file_path} {first_bin_out_folder}')
                os.system(f'cp -r {img_file_path} {second_bin_out_folder}')
        
        except Exception as e :
            print(f"[ERROR] {e} in file {img_file}")
            continue
    print('[INFO] Finished')

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Directory argument --> path to the imags' directory or images' zip file
    parser.add_argument('--directory', type=str, required=True )
    # model file path --> path to the model's pickle file 
    parser.add_argument('--model', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()
    # Run the main program 
    main(args.directory , args.model) 