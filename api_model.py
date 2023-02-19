import os 
import joblib

class ModelApi(object):

    def __init__(self, model_path='./output/models/') -> None:
        self.model_path = model_path


    def get_type_tag_pair(self):
        '''Return list of type and tag pair from all models'''
        type_tag_pair=[] # List for tag names

        try:                   
            for file in os.listdir(self.model_path):
                model_pkl_path = os.path.join(self.model_path , file)
                # Loading model object 
                with open(model_pkl_path, 'rb') as model_file:
                    model = joblib.load(model_file)
                    type_tag_pair.append((model['model_type'], model['tag']))
        except Exception as e:
            print (f'[ERROR] {e}: Failed getting models')
        
        return type_tag_pair


    def get_model_by_type_tag(self, model_type, tag):
        '''Return specific model dict based on given model_type and tag'''
        model=None

        try:                   
            for file in os.listdir(self.model_path):
                model_pkl_path = os.path.join(self.model_path , file)
                # Loading model object 
                with open(model_pkl_path, 'rb') as model_file:
                    model = joblib.load(model_file)
                    if model['model_type']==model_type and model['tag']==tag:
                        return model
        except Exception as e:
            print (f'[ERROR] {e}: Failed getting model')
        
        return model
        
    
    def get_models_dict(self):
        '''
        Returns models dictionary for all model pickle file

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
            # If it was a folder of all the models                      
            for file in os.listdir(self.model_path):
                if not file.endswith('pkl'):
                    # Not a model, skip it
                    continue
                model_pkl_path = os.path.join(self.model_path , file)
                model_name = os.path.splitext(file)[0]
                # Loading model object 
                with open(model_pkl_path, 'rb') as model_file:
                    models_dict[model_name] = joblib.load(model_file)
                        
            return models_dict

        except Exception as e:
            print (f'[ERROR] {e}: Failed getting models')
            return {}

# model_api = ModelApi()
# models_dict = model_api.get_models_dict('./output/models/')
# print (models_dict)