import argparse
import json
from cache_tag import TagCache
from api_model import ModelApi


def get_tag_cache_by_model(tag_cache_path, models_path):
    '''
    Get list of image file hash_id form tag cache based on tag value in models in given models_path 
    '''

    # Output placeholder
    model_tag_cache_pair = {}

    try:
        # Create tag cache object
        tag_cache = TagCache()
        # Create model api object
        model_api=ModelApi()

        # Getting models
        ret, models_dict = model_api.get_models_dict(models_path=models_path)
        '''
        Example stucture of models_dict
        {<model_name>: 
            {'classifier' : <model object>,
            'model_type' : <model type string>,
            'train_start_time' : <traing start time datetime object>
            'tag' : <tag string>
            }
        }
        '''

        # Get tags from models_dict
        for model in models_dict:
            # Get list of images (hash_IDs) based on each model's tag name
            hash_ids = tag_cache.get_hash_by_tag(db_path = tag_cache_path, tag = models_dict[model]['tag'])
            # Append list of hash IDs to the result dict
            model_tag_cache_pair[model] = hash_ids
        
        return model_tag_cache_pair

    except Exception as e:
        print (f'[ERROR] {e}: Getting image data failed')
        return None



if __name__=='__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', type=str, required=False, default='./output/tag_cache.sqlite')
    parser.add_argument('--tag_cache_path', type=str, required=False , default='./output/models/')
    args = parser.parse_args()

    # Path to directory containing model pickle files or specific model pickle file
    models_path = args.models_path
    # Path to tag cache file
    tag_cache_path=args.tag_cache_path

    # Get list of image file hash_id form tag cache based on tag value in models in given models_path 
    model_tag_cache_pair = get_tag_cache_by_model(tag_cache_path=tag_cache_path, models_path=models_path)
    
    # Print it
    print(json.dumps(model_tag_cache_pair, indent=2))

