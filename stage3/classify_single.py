import argparse
from classify_helper_functions import *

def main(
        image_path: str, 
        bins_number : int, 
        model_path : str, 
    ):

    image_tagging_folder = "./single_image_classification"
    bins_number = 10
    out_json = {}

    clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-B-32',pretrained= 'openai')
    bins_array = get_bins_array(bins_number) 
    blake2b_hash = file_to_hash(image_path)

    models_dict = create_models_dict(model_path)
    image_features = clip_image_features(image_path,clip_model,preprocess,device) # Calculate image features.

    classes_list = [] # a list of dict for every class 
    for model_name in models_dict:
        image_class_prob     = classify_image_prob(image_features,models_dict[model_name]) # get the probability list
        model_type, tag_name = get_model_tag_name(model_name) 
        tag_bin, other_bin   = find_bin(bins_array , image_class_prob) # get the bins 

        # Find the output folder and create it based on model type , tag name 
        tag_name_out_folder = make_dir([image_tagging_folder, f'{model_type}',f'{tag_name}',tag_bin])

        # Copy the file from source to destination 
        shutil.copy(image_path,tag_name_out_folder)

    classes_list.append({
                        'model_type' : model_type,
                        'tag_name'   : tag_name,
                        'tag_prob'   : image_class_prob[0]}
                        )

                                
    out_json[blake2b_hash] = {
            'hash_id'      : blake2b_hash,
            'file_path'    : image_path, 
            'classes'      : classes_list
            }
        
    save_json(out_json,image_tagging_folder) # save the .json file
    print("[INFO] Finished.")

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path'    , type=str, required=True)
    parser.add_argument('--model_path'        , type=str, required=True)
    parser.add_argument('--output_bins'  , type=int  ,required=False , default=5)

    args = parser.parse_args()

    # Run the main program 
    main(
        image_path    = args.image_path, 
        model_path     = args.model_path,
        bins_number    = args.output_bins
        )
