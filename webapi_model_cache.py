import argparse
import json
from flask import Flask, request

from api_model_cache import ModelCache


'''Model Cache Web API'''

# Server
app = Flask(__name__)

@app.route('/get_images_in_tag_score_range')
def get_model_dict():
    '''
    Returns models dictionary for model pickle file in given models_path
    Example http request: 
    http://127.0.0.1:8080/get_images_in_tag_score_range?model_name=model-ovr-logistic-regression-tag-not-pixel-art
    '''
    

    global model_cache
    '''
    Example stucture of files_dict

    {<hash_id>:
        {'file_name': <file name string>,
        'file_path': <model path string>,
        'file_type': <file type string>,
        'model_name': <file name of the model string>,
        'model_type': <type of the model string> ,
        'tag': <tag string>,
        'score':    <tag score float>
        }
    }
    '''

    try:
        # Argument extraction from URL query string
        _model_name = request.args.get('model_name')        
        _score_gte = request.args.get('score_gte')
        _score_lte = request.args.get('score_lte')
        # Checking model_name query string
        if _model_name:
            model_name = _model_name
        else:
            raise Exception ('Model name/score GTE/score LTE not specified')
        # Checking score GTE in query string
        if _score_gte:
            score_gte = _score_gte
        else:
            score_gte = 0.0
        # Checking score LTE in query string
        if _score_lte:
            score_lte = _score_lte
        else:
            score_lte = 1.0

        _, files_dict = model_cache.get_img_from_model_cache(model_name=model_name, score_gte=score_gte, score_lte=score_lte)

        print(json.dumps(files_dict, indent=2))
        return (json.dumps(files_dict, indent=2))

    except Exception as e:
        return (f'[ERROR] {e}: Getting image data failed')


if __name__=='__main__':

    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=False, default='0.0.0.0')
    parser.add_argument('--port', type=int, required=False , default=8080)

    args = parser.parse_args()

    HOST=args.host
    PORT_NUMBER=args.port

    try:
        # Initializing model cache object
        model_cache=ModelCache()

        # Run the server
        app.run(host=HOST, port=PORT_NUMBER)

    except Exception as e:
        print (e)
        