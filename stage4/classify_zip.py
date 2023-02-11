import argparse
import os
import sqlite3
import time
from zipfile import ZipFile
from PIL import Image
from stage4.classify_zip_helper_functions import *

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

def main(
        folder_path: str, 
        output_dir: str,
        json_file_path: str, 
        bins_number: int, 
        model_path: str, 
        ):

    zip_files = []
    global zips_info

    """main function to be running, calls other function.

    :param folder_path: path to the images' folder or archive file or single image file.
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
            zip_files.append(file)
    
    # Get the output folder path.
    if output_dir is None : 
        # Create base directory for ./output. Create the output directory name with time-stamp.
        image_tagging_folder = create_out_folder(base_dir = './output')
    else :
        image_tagging_folder =  create_out_folder(base_dir = output_dir)
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
    for file in tqdm(zip_files):
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
            
            # Appending zip archive name to file path
            #img_out_dict['file_path'] = f"{file}/{img_out_dict['file_path']}"
            out_json[img_out_dict['hash_id']] = img_out_dict

    # Save to output.json file
    save_json(out_json,image_tagging_folder) 


    '''
    Database writing
    Creating database and table for writing file info, model and classification result.
    '''
    
    db_out_dir = './output'
    #make sure result output path exists 
    os.makedirs(db_out_dir, exist_ok = True)
    DATABASE_NAME = '/zip_score_cache.sqlite'
    DATABASE_PATH = f'{db_out_dir}/{DATABASE_NAME}'
    
    print (DATABASE_PATH)
    def __delete_all_data_in_database():
        __delete_database()
        __create_database()

    def __create_database():
        cmd1 = '''CREATE TABLE zip_score_cache (
        file_name   TEXT    NOT NULL,
        file_path   TEXT            ,
        archive_path    TEXT        ,
        type            TEXT        ,
        n_img_content   INTEGER     ,
        hash_id     TEXT            ,
        model_name  TEXT            ,
        model_type  TEXT            ,
        model_train_date  TEXT      ,
        tag    TEXT                 ,
        tag_score    REAL    
        );
        '''

        db = sqlite3.connect(DATABASE_PATH)
        c = db.cursor()
        c.execute('PRAGMA encoding="UTF-8";')
        c.execute(cmd1)
        db.commit()
    
    def __delete_database():
        try:
            if(os.path.exists(DATABASE_PATH)):
                os.remove(DATABASE_PATH)
        except Exception as e:
            print(str(e))
            time.sleep(1)
            __delete_database()

    def __insert_file_into_database(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9):
        try:
            cmd = "insert into zip_score_cache(file_name, file_path, type, hash_id, model_name, model_type, model_train_date, tag, tag_score) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"', '"+arg5+"', '"+arg6+"', '"+arg7+"', '"+arg8+"', '"+arg9+"')"
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            if(str(e).find('lock') != -1 or str(e).find('attempt to write a readonly database') != -1):
                time.sleep(1)

    def __insert_zip_into_database(arg1, arg2, arg3, arg4):
        try:
            cmd = "insert into stage4(file_name, archive_path, type, n_img_content) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"')"
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            if(str(e).find('lock') != -1 or str(e).find('attempt to write a readonly database') != -1):
                time.sleep(1)

    __delete_all_data_in_database()

    print (f'[INFO] Writing to database table in {DATABASE_PATH}')

    '''
    Writing ZIP file info, model and classification result to the table by 
    '''
    for item in zips_info:
        file_name = os.path.basename(item[0])
        arch_path = item[0]
        n_content = item[1]
        __insert_zip_into_database(
            file_name,
            arch_path,
            os.path.splitext(file_name)[-1],
            str(n_content)
            )

    '''
    Writing file info, model and classification result to the table by 
    extracting data from out_json
    '''
    json_keys = list(out_json.keys())
    for key in json_keys:
        file_name = os.path.split(out_json[key]['file_path'])[-1]
        #file_path = os.path.split(out_json[key]['file_path'])[0]
        file_path = out_json[key]['file_path']
        hash_id = out_json[key]['hash_id']
        model_outs = out_json[key]['classifiers_output']
        for out in model_outs:
            model_name = out['model_name']
            model_type = out['model_type']
            model_train_date = out['model_train_date']
            tag = out ['tag']
            tag_score = out['tag_score']
            __insert_file_into_database(
                file_name,
                file_path,
                os.path.splitext(file_name)[-1],
                hash_id,
                model_name,
                model_type,
                model_train_date,
                tag,
                str(tag_score)
                )

    print("[INFO] Finished.")

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory'    , type=str, required=True , help="images directory or image file")
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
