import argparse
import json
from flask import Flask, request

from api_model import ModelApi


'''Model Access Web API'''

# Server
app = Flask(__name__)

@app.route('/get_models')
def get_models_dict():
    '''Returns models dictionary for model pickle file in given models_path'''

    global model_api
    global models_path
    '''
    Example stucture of models_dict
    {<model_name>: 
        {'classifier' : <model object>,
         'model_type' : <model type string>,
         'train_start_time' : <training start time datetime object>
         'tag' : <tag string>
        }
    }
    '''
    try:
        # Getting models dictionary
        models_dict = model_api.get_models_dict(models_path=models_path)
        if len(models_dict) > 0:
            # Creating new dict for return in HTTP request
            models_json = models_dict.copy()
            for model in models_json:
                # Create a new dict without classifier object
                models_json[model].pop('classifier')
                # Stringify the train_start_time datetime object
                models_json[model]['train_start_time'] = models_json[model]['train_start_time'].strftime('%Y-%m-%d, %H:%M:%S')
            
            print(json.dumps(models_json, indent=2))
            return (json.dumps(models_json, indent=2))
        else:
            return (f'[INFO]: No model exists...')

    except Exception as e:
        return (f'[ERROR] {e}: Failed getting available models...')


if __name__=='__main__':

    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=False, default='0.0.0.0')
    parser.add_argument('--port', type=int, required=False , default=8080)

    args = parser.parse_args()

    HOST=args.host
    PORT_NUMBER=args.port
    models_path = './output/models/'
    models_dict = {}

    try:
        # Initializing model api object
        model_api=ModelApi()
        app.run(host=HOST, port=PORT_NUMBER)

    except Exception as e:
        print (e)
        