import argparse
import os
import numpy as np
from classify_helper_functions import *

def main(
        folder_path: str, 
        output_dir : str, 
        json_file_path: str, 
        bins_number : int, 
        model_path : str, 
        ):
    """main function to be running, calls other function.

    :param folder_path: path to the images' folder or archive file.
    :type foldr_path: str
    :param output_dir: directory for the classification output, 
    :type output_dir: str
    :param json_file_path: .json file containing the hash , clip features and meta-data of the image.
    :type json_file_path: str
    :param bins_numbers: number of bins to divide the output into.
    :type bins_number: int
    :param model_path: path to the model's .pkl file or the directory of models' pickle files/
    :type model_path: str
    :rtype: None
    """
    #Check if it's an archived dataset 
    if folder_path.endswith('.zip'): 
        folder_path = unzip_folder(folder_path) # will be unzipped in the current directory of the script.
    
    # Clean the directoy (converts every .GIF to .PNG).
    clean_directory(folder_path)

    # Get the output folder path.
    if output_dir is None : 
        # Create the output directory name with time-stamp.
        image_tagging_folder = create_out_folder()
    else :
        image_tagging_folder = output_dir
    print(f"[INFO] Output folder {image_tagging_folder}")
    
    # Load the .json file.
    metadata_json_obj = load_json(json_file_path)
    
    if metadata_json_obj is None:
        print("[WARNING] No .json file loaded, calculating embeddings for every image.")

    # Get CLIP model, to calculate CLIP embeddings if it's not in .json metadata file.
    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-B-32',pretrained= 'openai')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path,'output','models') if model_path is None else model_path
    models_dict = create_models_dict(model_path)
    
    bins_array = get_bins_array(bins_number) 

    out_json = {} # a dictionary for classification scores for every model.
    # Loop through each image in the folder.
    for img_file in tqdm(os.listdir(folder_path)):

        try:    
            image_file_path = os.path.join(folder_path, img_file)
            blake2b_hash = file_to_hash(image_file_path)
    
            try : 
                image_features = np.array(metadata_json_obj[blake2b_hash]["embeddings_vector"]).reshape(1,-1) # et features from the .json file.
            except KeyError:
                image_features = clip_image_features(image_file_path,clip_model,preprocess,device) # Calculate image features.

                classes_list = [] # a list of dict for every class 
                # loop through each model and find the classification of the image.
                for model_name in models_dict:
                    try :
                                        
                        image_class_prob     = classify_image_prob(image_features,models_dict[model_name]) # get the probability list
                        model_type, tag_name = get_model_tag_name(model_name) 
                        tag_bin, other_bin   = find_bin(bins_array , image_class_prob) # get the bins 

                        # Find the output folder and create it based on model type , tag name 
                        tag_name_out_folder = make_dir([image_tagging_folder, f'{model_type}',f'{tag_name}',tag_bin])
                        
                        # Copy the file from source to destination 
                        shutil.copy(image_file_path,tag_name_out_folder)

                        classes_list.append({
                                            'model_type' : model_type,
                                            'tag_name'   : tag_name,
                                            'tag_prob'   : image_class_prob[0]})

                    # Handles any unknown/unexpected errors for an image file.
                    except Exception as e  :
                        print(f"[ERROR] {e} in file {img_file} in model {model_name}")
                        continue
    
                out_json[blake2b_hash] = {
                            'hash_id'                 : blake2b_hash,
                            'file_path'               : image_file_path,
                            'classifiers_output'      : classes_list 
                            }
                                                
        except Exception as e :
            print(f"[ERROR] {e} in file {img_file}")
            continue


    save_json(out_json,image_tagging_folder) # save the .json file
    print("[INFO] Finished.")

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory'    , type=str, required=True)
    parser.add_argument('--output'       , type=str, required=False , default=None)
    parser.add_argument('--metadata_json', type=str, required=False , default=None)
    parser.add_argument('--model'        , type=str, required=False, default=None)
    parser.add_argument('--output_bins'  , type=int  ,required=False , default=10)

    args = parser.parse_args()

    # Run the main program 
    main(
        folder_path    = args.directory, 
        output_dir     = args.output, 
        json_file_path = args.metadata_json, 
        bins_number    = args.output_bins,
        model_path     = args.model
        ) 
