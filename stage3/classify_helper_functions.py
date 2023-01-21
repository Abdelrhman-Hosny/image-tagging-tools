import sys
sys.path.insert(0, './')
from PIL import Image
import os 
import torch
import shutil
import patoolib
from typing import List , Union
import open_clip
from tqdm import tqdm 
import joblib
import json
import hashlib
import datetime
import numpy as np
import io
import datetime
#from model_pytorch_logistic_regression_classifier import LogisticRegressionPytorch


def save_json(
              dict_obj,
              out_folder
                ):
    """saves a dict into as a .json file in a specific directory.
    
    :param dict_obj: a dictionary which will be converted to .json file.
    :type dict_obj: dict.
    :param out_folder: output folder for the .json file.
    :type out_folder: str
    :rtype: None
    """
    # Serializing json
    json_object = json.dumps(dict_obj, indent=4)    
    # Writing to output folder
    with open(os.path.join(out_folder , "output.json"), "w") as outfile:
        outfile.write(json_object)

def get_model_tag_name(model_file_name: str):
    """ get the model type and tag name given model .pkl file name.

    :param model_file_name: name of the .pkl file of the model.
    :type model_file_name: str.
    :return: a tuple of (model type , tag name)
    :rtype: Tuple[str] 
    """
    # Get model name and tag from model's dict keys.
    return model_file_name.split('-tag-')[0].split('model-')[1] , model_file_name.split('-tag-')[1] 


def find_bin(
             bins_arr,
             prob
             ):
    """ find the bin value for tag class and other class
    
    :param bins_arr: Numpy array holding the bins values.
    :type bins_arr:  NdArray
    :param prob: value of the probability to be in a certain tag..
    :type prob:  float.
    :returns: a tuple of tag bin value and other bin value.
    :rtype: Tuple[str]
    """
    return str(bins_arr[np.absolute(bins_arr-prob).argmin()]) , str(bins_arr[np.absolute(bins_arr-(1-prob)).argmin()])


def get_bins_array(bins_number):
    """generate array of bins using the number of bins choosen
    
    :param bins_number: number of bins to be generated.
    :type bins_number: int
    :returns: a Numpy array of the bins values.
    :rtype: Numpy Array
    """
    num = 1.0 / bins_number
    return np.array([round(i , 3) for i in np.arange(num , 1.0+num , num)])


def classify_image_prob(
                    image_features,
                    model,
                    torch_model=False
                    ):
    """calculate the classification prediction giving model and CLIP
    image features.
    
    :param image_features: features for the image from CLIP (CLIP embeddings).
    :type imagE_features:  [1,512] Numpy array
    :param model: Classification model from the .pkl file.
    :type model: Object.
    :param torch_model: to determin if the the moel is or not a Pytorch model.
    :type torch_model: bool
    :returns: value of probbility of a certain image to be in a certain tag.
    :rtype: float
    """
    if not torch_model:
      prob = model.predict_proba(image_features)[0][0]
    else:
      with torch.no_grad():
        prob = model(torch.from_numpy(image_features.reshape(1,-1).astype(np.float32))).detach().numpy()[0][0]
        prob = 1 - prob
    return prob


def load_json(json_file_path:str):
    """ Takes a path for json file then returns a dictionary of it.

        :param json_file_path: path to the json file.
        :type json_file_path: str
        :returns: dictionary of the json file.
        :rtype: dict
    """
    try :

        with open(json_file_path, 'rb') as json_obj:
            json_dict = json.load(json_obj)
        json_obj.close()

    except Exception as e : # handles any exception of the json file
        print("[ERROR] Probem with the json file")
        return None
    
    return json_dict

def create_out_folder(base_dir = './'):
    """creates output directory for the image classification task.
    
    :returns: path to the output directory.
    :rtype: str
    """
    timestamp = datetime.datetime.now() 
    # RV: Adding base directory
    image_tagging_folder_name = os.path.join(base_dir, f'tagging_output_{timestamp.year}_{timestamp.month}_{timestamp.day}_{timestamp.hour}_{timestamp.minute}')
    return make_dir(image_tagging_folder_name)

def compute_blake2b(image: Image.Image): 
    """compute the BLAKE2b of a PIL image. 

    :param image: The PIL image to compute its BLAKE2b
    :type image: PIL.Image.Image
    :returns: the BLAKE2b str of the given image. 
    :rtype: str
    """
    
    return hashlib.blake2b(image.tobytes()).hexdigest()


def make_dir(dir_names : Union[List[str] , str]):
    """takes a list of strings or a string and make a directory based on it.
    :param dir_name: the name(s) which will be the path to the directory.
    :type dir_name: Union[List[str] , str]
    :returns: a path to the new directory created 
    :rtype: str
    """
    if type(dir_names) == str:
        if dir_names.strip() == "":
            raise ValueError("Please enter a name to the directory")
   
        os.makedirs(dir_names , exist_ok=True)
        return dir_names
  
    elif type(dir_names) == list and len(dir_names) == 0:
        raise ValueError("Please enter list with names")
  
    elif type(dir_names) == list and len(dir_names) == 1:
        os.makedirs(dir_names[0] , exist_ok=True)
        return dir_names[0]
  
    final_dir = os.path.join(dir_names[0] , dir_names[1])
    for name in dir_names[2:]:
        final_dir = os.path.join(final_dir , name)

    os.makedirs(final_dir , exist_ok=True)
    return final_dir


def unzip_folder(folder_path :str):
    """takes an archived file path and unzip it.
    :param folder_path: path to the archived file.
    :type folder_path: str
    :returns: path of the new exracted folder 
    :rtype: str
    """
    dir_path  = os.path.dirname(folder_path)
    # Use the file name as it is
    #file_name = os.path.basename(folder_path).split('.zip')[0]
    file_name = os.path.basename(folder_path)
    #os.makedirs(dir_path , exist_ok=True)
    
    # RV
    #output_path = f"{file_name}-decompressed-tmp"
    output_path = f"{file_name}"
    output_directory = os.path.join('./outputs/tmp' , output_path)
    #make sure the output dir is found or else create it. 
    os.makedirs(output_directory, exist_ok = True)

    print("[INFO] Extracting the archived file...")
    patoolib.extract_archive(folder_path, outdir=output_directory)
    print("[INFO] Extraction completed.")
    return output_directory

    #return os.path.join(dir_path, file_name)

def get_clip(clip_model_type : str = 'ViT-B-32' ,
             pretrained : str = 'openai'):
      """initiates the clip model, initiates the device type, initiates the preprocess
      :param clip_model_type: type of the CLIP model. 
      :type clip_model_type: str
      :param pretrained: pretrained name of the model.
      :type pretrained: str
      :returns: clip model object , preprocess object , device object
      :rtype: Object , Object , str
      """
      # get clip model from open_clip
      clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_type,pretrained=pretrained)
      device = "cuda" if torch.cuda.is_available() else "cpu"

      return clip_model , preprocess , device

# image = Image.open(io.BytesIO(bytes))

def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a fake file stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file as a argument, passing a bytes io ins
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def create_models_dict(models_path:str):
    """take the path of the models' folder, load all of them in one dict
    :param models_path: path to the models pickle files path
    :type models_path: str
    :returns: dictionary contains all the models with their names
    :rtype: Dict
    """
    models_dict = {} # Dictionary for all the models objects
    
    if models_path.endswith('.pkl'):     # If it was just a single model file.
        model_name = os.path.basename(models_path).split('.pkl')[0]

        # Loading model object 
        with open(models_path, 'rb') as model:
          # model is a dict with the following structure
          # {'classifier': <classifier>, 'model_type': <model_type>, 'train_start_time': <train_start_time>, 'tag': <tag_name>}
          models_dict[model_name] = joblib.load(model)
        
    else:                               # If it was a folder of all the models.
      for model_file in os.listdir(models_path):
          if not model_file.endswith('pkl'):
              continue

          model_pkl_path = os.path.join(models_path , model_file)
          model_name = model_file.split('.pkl')[0]
      
          # Loading model object 
          with open(model_pkl_path, 'rb') as model:
            # model is a dict with the following structure
            # {'classifier': <classifier>, 'model_type': <model_type>, 'train_start_time': <train_start_time>, 'tag': <tag_name>}
            models_dict[model_name] = joblib.load(model)

    return models_dict


def create_mappings_dict(mappings_path:str):
    """take the path of the mappings' folder, load all of them in one dict
    
    :param mappings_path: path to the models pickle files path
    :type models_path: str
    :returns: dictionary contains all the models with their names
    :rtype: Dict
    """
    mappings_dict = {} # Dictionary for all the models objects
    for mapping_file in tqdm(os.listdir(mappings_path)):
        if not mapping_file.endswith('json'):
            continue

        mapping_file_path = os.path.join(mappings_path , mapping_file)
        model_name = mapping_file.split('.json')[0]

        with open(mapping_file_path, 'rb') as mapping_obj:
            mappings_dict[model_name] = json.load(mapping_obj)
        mapping_obj.close()

    return mappings_dict


def clip_image_features(
                        image_path : Union[str,bytes],
                        model, 
                        preprocess,
                        device:str
                        ):
    """ returns the features of the image using OpenClip

        :param image_path: path to the image to get it's features.
        :type iamge_path: str
        :param model: CLIP model object to get the features with.
        :type model: CLIP model Object
        :param preprocess: preprocess methods will be applied to the image.
        :type preprocess: CLIP preprocess object.
        :param device: device which will be used in calculating the features.
        :type device: str
        :returns: CLIP features of the image.
        :rtype: [1,512] Numpy array.
    """    
    with torch.no_grad():

        # check if it's a byte object.
        if isinstance(image_path, (bytes, bytearray)):
          img_obj = Image.open(io.BytesIO(image_path))

        elif image_path.lower().endswith('.gif'): 
          img_obj = convert_gif_to_image(image_path)  
        
        else :
          img_obj = Image.open(image_path)

        image = preprocess(img_obj).unsqueeze(0).to(device)
        return model.encode_image(image).detach().numpy()


def clean_file(file_path):
  """This function takes a file path and see if it is supported or not. 
     suported files : ['.gif','.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    
    :param file_path: path of the file to work with 
    :type file_path: str
  """
  # if it's not gif and not a supprted file type then remove it 
  # RV Adding zip, Disables this
  # if not file_path.lower().endswith(('.zip', '.gif','.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
  #   os.remove(file_path)
  #   print(f'[Removing] {file_path}')
  #   return 
  pass


def convert_gif_to_image(gif_path: str):
  """gets the first frame of .gif file and returns it.

  :param gif_path: path to the GIF file.
  :type gif_path: str
  :returns: image of the first frame of the .gif file.
  :rtype: Image.Image
  """
  im = Image.open(gif_path)
  im.seek(0)
  return im 


def list_models(model_folder_path: str):
  """Listing all the models from a model's folder.

  :param model_folder_path: path to the folder having all the classification models.
  :type model_folder_path: str
  :rtype: None
  """
  models_dict = create_models_dict(model_folder_path)
  models = {}
  for model_name in models_dict:
    model_type, tag_name = get_model_tag_name(model_name) 
    
    if model_type not in models.keys():
      models[model_type] = [tag_name]
      continue
    models[model_type].append(tag_name)

  for model_type in models:
    for tag_name in models[model_type]:
      print(f"model-type = {model_type}, tag= {tag_name}")



def generate_model_path(
                        model_folder_path: str,
                        model_type: str,
                        tag_name: str
                        ):
  """generating model path from just model type and model tag name.

  :param model_folder_path: path to all the models folder.
  :type model_folder_path: str
  :param model_type: type of the model ('ovr-logistic-regression','ovr-svm')
  :type model-_type: str
  :param tag_name: name of the tag of the model.
  :type tag_name: str
  :returns: a path to the model's .pkl file.
  :rtype: str 
  """
  pkl_file_path = os.path.join(model_folder_path,f'model-{model_type}-tag-{tag_name}.pkl')
  return pkl_file_path if os.path.exists(pkl_file_path) else None 

def classify_single_model_to_float(
                                  image_file_path: str,
                                  model,
                                  model_name: str,
                                  image_features,
                                  torch_model = False
                                  ):
    """classifying image file using only a single model object.

    :param image_file_path: path to the image file.
    :type image_file_path: str
    :param model: model's object.
    :type model: Object (SVM or LogisticRegression)
    :param model_name: name of the input model.
    :type model_name: str.
    :param bins_array: array including the list of the bins for classification.
    :type bins_array: List[float]
    :param image_features: CLIP embedding features of the input image.
    :type image_features: NdArray.
    """
    try :
      return classify_image_prob(image_features,model ,torch_model=torch_model) # get the probability list

    except Exception as e  :
        print(f"[ERROR] {e} in file {os.path.basename(image_file_path)} in model {model_name}")
        return None

def classify_to_float(
                      image_path: Union[str,bytes],
                      model_name: str,
                      tag_name: str
                      ):
  
  """ returns the float value of the output probability from a given model"""
  
  clip_model, preprocess , device = get_clip()
  
  model_path = generate_model_path('./image-tagging-tools/output/models',model_type= model_name ,tag_name= tag_name)
  model_dict = create_models_dict(models_path=model_path)

  image_features = clip_image_features(image_path,clip_model,preprocess,device) # Calculate image features.

  return classify_image_prob(image_features,model_dict[f"model-{model_name}-tag-{tag_name}"] ,torch_model=("torch" in model_name))

def classify_single_model_to_bin(
                                  image_file_path: str,
                                  model,
                                  model_name: str,
                                  image_features,
                                  bins_array: List[float],
                                  image_tagging_folder: str,
                                  torch_model = False
                                  ):
    """classifying image file using only a single model object.

    :param image_file_path: path to the image file.
    :type image_file_path: str
    :param model: model's object.
    :type model: Object (SVM or LogisticRegression)
    :param model_name: name of the input model.
    :type model_name: str.
    :param bins_array: array including the list of the bins for classification.
    :type bins_array: List[float]
    :param image_features: CLIP embedding features of the input image.
    :type image_features: NdArray.
    """
    try :

      # Take the classifier from model
      classifier = model['classifier']

      image_class_prob     = classify_image_prob(image_features, classifier,torch_model=torch_model) # get the probability list
      model_type, tag_name = get_model_tag_name(model_name) 
      tag_bin, other_bin   = find_bin(bins_array , image_class_prob) # get the bins 

      # Find the output folder and create it based on model type , tag name 
      tag_name_out_folder = make_dir([image_tagging_folder, f'{model_type}',f'{tag_name}',tag_bin])
    
      # Copy the file from source to destination 
      shutil.copy(image_file_path,tag_name_out_folder)

      return  { 'model_name' : model_name,
                'model_type' : model_type,
                'model_train_date' : model['train_start_time'].strftime('%Y-%m-%d, %H:%M:%S'),
                'tag'   : tag_name,
                'tag_score'   : image_class_prob
              }

    except Exception as e  :
        print(f"[ERROR] {e} in file {os.path.basename(image_file_path)} in model {model_name}")
        return None
      

def classify_to_bin(
                    image_file_path: str,
                    models_dict: dict,
                    metadata_json_obj: dict,
                    image_tagging_folder: str,
                    bins_array: List[float],
                    clip_model,
                    preprocess,
                    device
                    ):
  """classification for a single image through all the models.

  :param image_file_path: path to the image will be classified.
  :type image-file_path: str
  :param models_dict: dictionary of all available classification models.
  :type models_dict: dict
  :param metadata_json_object: a dictioanry loaded from the .json file.
  :type  metadata_json_object: dict
  :param image_tagging_folder: path to the image tagging folder (output folder)
  :type image_tagging-folder: str
  :param bins_array: array of all available bins for classification.
  :type bins_array: List[float]
  :param clip_model: CLIP model object for getting the image features.
  :type clip_model. CLIP
  :param preprocess: preprocessing object for images before getting into CLIP.
  :type preprocess: Object.
  :param device: device name
  :type device: str
  """
  try:    

    blake2b_hash = file_to_hash(image_file_path)
    
    try : 
        image_features = np.array(metadata_json_obj[blake2b_hash]["embeddings_vector"]).reshape(1,-1) # et features from the .json file.
    except KeyError:
        image_features = clip_image_features(image_file_path,clip_model,preprocess,device) # Calculate image features.

    classes_list = [] # a list of dict for every class 

    # loop through each model and find the classification of the image.
    for model_name in models_dict:
      model = models_dict[model_name]
      torch_model = 'torch' in model_name
      # model is a dict with the following structure
      # {'classifier': <classifier>, 'model_type': <model_type>, 'train_start_time': <train_start_time>, 'tag': <tag_name>}
      model_result_dict = classify_single_model_to_bin(
                                                        image_file_path,
                                                        model,
                                                        model_name,
                                                        image_features,
                                                        bins_array,
                                                        image_tagging_folder,
                                                        torch_model=torch_model
                                                        )
      if model_result_dict is None:
        continue

      classes_list.append(model_result_dict)

    return {'hash_id'  :  blake2b_hash,
            'file_path': image_file_path,
            'classifiers_output': classes_list}

  except Exception as e :
    print(f"[ERROR] {e} in file {os.path.basename(image_file_path)}")
    return None 


def file_to_hash(file_path: str):
    """converts file (.gif or else) into blake 2b hash

    :param file_path: file path to be converted.
    :type file_path: str
    :returns: blake 2b hash of the file.
    :rtype: str
    """
    if file_path.lower().endswith('.gif'): # If it's GIF then convert to image and exit 
      try : 
        return compute_blake2b(convert_gif_to_image(file_path))
      except Exception as e:
        print(f"[ERROR]  cannot compute hash for {file_path} , {e}")
        return None 
    return compute_blake2b(Image.open(file_path))

def empty_dirs_check(dir_path : str):
      """ Checking for empty directory"""
      for dir in os.listdir(dir_path):
        sub_dir = os.path.join(dir_path, dir)

        # Check for directory only
        if os.path.isdir(sub_dir):
            if len(os.listdir(sub_dir)) == 0:
              # Empty folder
              print(f'[Warning] Empty folder found. Ignoring it: {sub_dir}')
              continue

def from_prob_to_bin(prob:float):
      """Divide the output to 10 bins
      :param prob: probability of an image being in a certain class.
      :type prob: float
      """   
      if 0.0 <= prob <= 0.1 : 
        return '0.1'
      elif 0.1 < prob <= 0.2 : 
        return '0.2'
      elif 0.2 < prob <= 0.3:
          return '0.3'
      elif 0.3 < prob < 0.4 : 
        return '0.3'
      elif 0.4 <= prob < 0.5 : 
        return '0.4'
      elif 0.5 <= prob < 0.6 : 
        return '0.5'
      elif 0.6 <= prob < 0.7 : 
        return '0.6'
      elif 0.7 <= prob < 0.8 : 
        return '0.7'
      elif 0.8 <= prob < 0.9 : 
        return '0.8'
      elif 0.9 <= prob < 1.0 : 
        return '0.9'
      elif prob == 1.0 : 
        return '1.0' 

def classify_image_bin(
                       image_path,
                       classifier,
                       mapper,
                       clip_model,
                       preprocess,
                       device
                       ):
    """Returns the string of "tag/class" or "other" 

       :param image_path: path to the image which will be classified.
       :type image_path: str
       :param classifier: classifier object.
       :type classifier: Classifier .pkl file Object
       :param mapper: mapping object to define which tag is index 0 and which tag is index 1.
       :type mapper: JSON object
       :param clip_model: CLIP model object to get the features with.
       :type clip_model: CLIP model Object.
       :type preprocess: CLIP preprocess object.
       :param device: device which will be used in calculating the features.
       :type device: str
       :returns: dictinary including "tag name" : "bin value" , ex: "pos-pixel-art" : "0.7" 
       :rtype: dict
    """
    image_features   = clip_image_features(image_path , clip_model , preprocess , device)
    probs = classifier.predict_proba(image_features)
    first_class_bin  = from_prob_to_bin(probs[0][0])
    second_class_bin = from_prob_to_bin(probs[0][1])
    return  { 
              mapper['0'].strip() : first_class_bin , 
              mapper['1'].strip() : second_class_bin
            }

class FolderFilter:

      def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.allowed_extension = ('.gif','.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


      def __accepted_file(self, file_path):
        """A test of a file to see if it matches a group of criteria or not"""

        return os.path.isfile(file_path) and os.path.basename(file_path).endswith(self.allowed_extension)
            
      
      def run(self):
        """ Loops over a folder and returns a list of the path it should classifiy/pre-compute
        this is just a stage for future stages. Instead of deleting, ignoring.
        """        
        files_list = [] # a list of all files in the folder after filtering.
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)

            if self.__accepted_file(file_path):
              files_list.append(file_path)
        
        return files_list

      





