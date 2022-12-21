import argparse
import os
import numpy as np
from zipfile import ZipFile
from PIL import Image
from classify_helper_functions import *

zips_info = []

def zip_gen(zip_file):
    '''Image generator for zip file'''
    
    global zips_info

    print (f'[INFO] Working on ZIP archive: {zip_file}')
    
    with ZipFile(zip_file) as archive:

        '''Getting archive details'''
        # Check the number of content (image file)
        entries = archive.infolist()
        n_content =  len([content for content in entries if content.is_dir() ==False])
        # Appending to the list to be writen to database table
        zips_info.append([zip_file, n_content])

        for entry in entries:
            # Do for every content in the zip file
            if not entry.is_dir():
                
                with archive.open(entry) as file:

                    if entry.filename.lower().endswith(('.zip')):
                        # Another zip file found in the content.
                        print (f'[INFO] Working on ZIP archive: {zip_file}/{entry.filename}')
                        # Process the content of the zip file
                        with ZipFile(file) as sub_archive:

                            '''Getting archive details'''
                            # Check the number of content
                            sub_entries = sub_archive.infolist()
                            n_content =  len([content for content in sub_entries if content.is_dir() ==False])
                            # Appending to the list to be writen to database table
                            zips_info.append([f'{zip_file}/{entry.filename}', n_content])

                            for sub_entry in sub_entries:
                                with sub_archive.open(sub_entry) as sub_file:
                                    try:
                                        img = Image.open(sub_file)
                                        img_file_name = f'{zip_file}/{sub_entry.filename}'
                                        print (f' Processing: {img_file_name}')
                                        yield (img, img_file_name)
                                    except:
                                        print (f'[WWARNING] Failed to open {os.path.join(zip_file, sub_entry.filename)}')
                                        continue
                    else:
                        # Should be image file. Read it.
                        try:
                            img = Image.open(file)
                            img_file_name = entry.filename
                            print (f' Processing: {img_file_name}')
                            yield (img, img_file_name)
                        except:
                            print (f'[WARNING] Failed to open {entry.filename}')
                            continue

def main(folder_path: str = 'images', 
        output_dir: str = 'output', 
        json_file_path: str = 'input-metadata.json', 
        bins_number: int = 5, 
        model_path: str = 'models', 
        ):

    zips = []
    global zips_info

    if not os.path.isfile(folder_path):
        # Check for empty dirs
        empty_dirs_check(folder_path)
        # Placeholder for data file names
        img_files_list = []
        # Walking thru files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                img_files_list.append(os.path.join(root, file))
    else:
        img_files_list = [folder_path]
    
    # Selecting zip files only
    for file in img_files_list:
        if file.lower().endswith(('.zip')):
            zips.append(file)
    
    # Get the output folder path.
    if output_dir is None : 
        # Create base directory for ./output. Create the output directory name with time-stamp.
        image_tagging_folder = create_out_folder(base_dir = './output')
    else :
        image_tagging_folder = output_dir
    print(f"[INFO] Output folder {image_tagging_folder}")
    
    # Load the .json file.
    metadata_json_obj = load_json(json_file_path)
    
    if metadata_json_obj is None:
        print("[WARNING] No .json file loaded, calculating embeddings for every image.")

    # Get CLIP model, to calculate CLIP embeddings if it's not in .json metadata file.
    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-B-32',pretrained= 'openai')
    model_path  = os.path.join('output','models') if model_path is None else model_path
    models_dict = create_models_dict(model_path)
    bins_array  = get_bins_array(bins_number) 

    out_json = {} # a dictionary for classification scores for every model.
        
    # Loop through each zip file.
    for file in tqdm(zips):
        # Generating images
        for img, img_file_name in tqdm(zip_gen(file)):
            # Classify
            img_out_dict = classify_to_bin(
                                            img,
                                            img_file_name,
                                            models_dict,
                                            metadata_json_obj,
                                            image_tagging_folder,
                                            bins_array,
                                            clip_model,
                                            preprocess,
                                            device
                                        )
            if img_out_dict is None:
                continue
            out_json[img_out_dict['hash_id']] = img_out_dict

    # Save to output.json file
    save_json(out_json,image_tagging_folder) 

    print("[INFO] Finished.")

if __name__ == '__main__':
    # Create the parser
    # Run the main program 
    main() 
