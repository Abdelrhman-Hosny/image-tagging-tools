import os 
import joblib

class ModelApi(object):

    def get_models_dict(self, models_path):
        '''
        Returns models dictionary for model pickle file in given models_path.

        Example stucture of models_dict
        {<model_name>: 
            {'classifier' : <model object>,
            'model_type' : <model type string>,
            'train_start_time' : <training start time datetime object>
            'tag' : <tag string>
            }
        }
        '''

        models_dict={} # Dictionary for all the models objects
        
        try:
            if os.path.isfile(models_path):
                if models_path.endswith('.pkl'):
                    # If it was just a single model file    
                    model_name = os.path.splitext(os.path.split(models_path)[-1])[0]
                    # Loading model object 
                    with open(models_path, 'rb') as model:
                        models_dict[model_name] = joblib.load(model)
        
            else:
                # If it was a folder of all the models                      
                for model_file in os.listdir(models_path):
                    if not model_file.endswith('pkl'):
                        # Not a model, skip it
                        continue
                    model_pkl_path = os.path.join(models_path , model_file)
                    model_name = os.path.splitext(model_file)[0]
                    # Loading model object 
                    with open(model_pkl_path, 'rb') as model:
                        models_dict[model_name] = joblib.load(model)
                        
            return True, models_dict

        except Exception as e:
            print (f'[ERROR] {e}: Failed getting models')
            return False, models_dict

# model_api = ModelApi()
# models_dict = model_api.get_models_dict('./output/models/')
# print (models_dict)