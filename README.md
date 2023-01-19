

### Pipeline Colab Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/image-tagging-tools/blob/main/image-tagging-tools-examples.ipynb)

You can run the pipeline on google colab using the following [link](https://colab.research.google.com/github/kk-digital/image-tagging-tools/blob/main/image-tagging-tools-examples.ipynb)

## Installation
All what's needed to start using the pipeline locally is to have python 3.9+ then run the following command

```sh
pip install -r ./requirements.txt
```

# Stage 1: Image Dataset Processor

> A standalone tool for processing dataset of tagged images and calculating its metadata and the CLIP embeddings. 

## Tool Description

Process a directory of images (paths to directory of images or an archived dataset) and computes the images metadata along with its CLIP embeddings and writes the result into a JSON file in specified output folder.

## Example Usage

* Process a tagged dataset (in this example is in `./dataset` folder) and save the output into `./output` folder. In addition, the SQLite database named `dataset_cache.sqlite` with table named `dataset_cache` containing file name, hash and file path for dataset images will be created in the `./output` folder. 

```sh
python ./stage1/ImageDatasetProcessor.py --input_folder=./dataset 
```

* Process a non-tagged dataset (in this example is in `./dataset` folder) and save the output into `./output/clip-cache` folder.  In addition, the SQLite database named `dataset_cache.sqlite` with table named `dataset_cache` containing file name, hash and file path for dataset images will be created in the `./output/clip-cache` folder.

```sh
python ./stage1/ImageDatasetProcessor.py --input_folder=./dataset --tagged_dataset=False
```

## CLI Arguments

* `input_folder` _[string]_ - _[required]_ - path to the directory containing sub-folders of each tag.
* `output_folder` _[string]_ - _[optional]_ - path to the directory where to save the files into it.
* `tagged_dataset` _[bool]_ - _[optional]_ - the dataset to process is a tagged dataset such that each each parent folder name is the tag of the images contained within it, default is `True`

* `clip_model` _[str]_ - _[optional]_ CLIP model to be used, default is `'ViT-B-32'`

* `pretrained` _[str]_ - _[optional]_ -  the pre-trained model to be used for CLIP, default is `'openai'`

* `batch_size` _[int]_ - _[optional]_ -  number of images to process at a time, default is `32`. 
* `num_threads` _[int]_ - _[optional]_ - the number to be used in this process, default is `4`

* `device` _[str]_ - _[optional]_ -  the device to be used in computing the CLIP embeddings, if `None` is provided then `cuda` will be used if available, default is `None`


# Stage 2: Train Classifier


> A script for training classification models based on `input-tag-to-image-hash-list.json` file and `input-metadata.json` file.

## Tool Description

Given a `metadata` json file containing embeddings for images and `tag-to-image-hash` json file containing images' hash with tags, the script start to make for every tag two binary classification models and save it in output folder.

## Example Usage

```
python ./stage2/train.py --metadata_json  './output/input-metadata.json' --tag_to_hash_json './output/input-tag-to-image-hash-list.json'
```

> Note that if the `output` is not created the script automatically creates it for you. 


> Note that if the `test_per` is not created the script will make test set ~= 10% of the dataset.

Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Parameters

* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file. 
* `tag_to_hash_json` _[string]_ - _[required]_ - The path to tag-to-hash json file. 

* `output` _[string]_ - _[optional]_ - The path to the output directory.
* `test_per` _[float]_ - _[optional]_ - The percentage of the test images from the dataset, default = 0.1 


# Stage 3: Classify Data

> A script for classification models inference given images' `directory` and `metadata_json` .json file. This stage does not process .zip (archived) file.



## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder, the script start to loop over every image and make the classification for it using every binary classification model. In addition, the SQLite database named `stage3.db` (containing file name, file path, hash, model type, tag name and tag score for given images) will be created in the `output` folder in the root directory. 

## Example Usage

```
python ./stage3/classify.py --metadata_json './output/input-metadata.json' --directory ‘./output/images_directory’
```

```
python ./stage3/classify.py --metadata_json './input-metadata.json' \
                            --directory ‘/src/to/dir/images_directory’
                            --output ‘./classification_output’
                            --output_bins 10
                            --model ‘./output/models’

```



> Note that if the `output` is not created the script automatically creates it for you. 

> Note that if the `model` is not created the script automatically uses models in [outputs/models](outputs/models/)

Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Parameters

* `directory` _[string]_ - _[required]_ - The path to the images folder or images .zip file. 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.


# Stage 4: Classify Data (for ZIP Archives)

> A script for classification models inference given images' `directory` and `metadata_json` .json file. This stage only process .zip (archived) file. It is used if the image data specified in `directory` argument is either in .zip archived format or containing images archived in .zip format. 

## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder, the script start to loop over every image and make the classification for it using every binary classification model. In addition, the SQLite database named `stage4.db` (containing file name, file path, archive path, type of file, hash, model type, tag name and tag score for given images) will be created in the `output` folder in the root directory. 

## Example Usage

```
python ./stage4/classify_zip.py --metadata_json './output/input-metadata.json' --directory ‘./output/images_directory.zip’
```

```
python ./stage4/classify_zip.py --metadata_json './input-metadata.json' \
                            --directory ‘/src/to/dir/images_directory.zip’
                            --output ‘./classification_output’
                            --output_bins 10
                            --model ‘./output/models’

```



> Note that if the `output` is not created the script automatically creates it for you. 

> Note that if the `model` is not created the script automatically uses models in [outputs/models](outputs/models/)

Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Parameters

* `directory` _[string]_ - _[required]_ - The path to the images folder or images .zip file. 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.


# File Cache Module
> A tool to create file cache in the form of SQLite database, add data to and fetch from it. File cache contains attributes for each file in given directory or folder. The attributes for each file are represented by the following fields in file cache SQLite database table: `hash_id`, `file_name`, `path`, `type`, `is_archive`, `n_content` and `container_archive`.

## Module Description
The file cache module defined in `cache_file.py` contains the class definition with the following functions:

* _class_  `cache_file`.__`FileCache`__ - A class to construct file cache object.
* __`create_file_cache`__(_`out_dir = './output'`_, _`db_name = 'file_cache.sqlite'`_) - Method to create file cache database with default name `file_cache.sqlite` and default location in project `./output` directory. The file cache database will not be created if the database with same name path already exist.
* __`add_folder_to_file_cache`__(_`data_dir`_, _`out_dir = './output'`_, _`db_name = 'file_cache.sqlite'`_) - Method to add image files contained in a folder / directory specified in `data_dir` to file cache database specified in `db_name` at location `out_dir` directory.
* __`get_random_hash`__(_`db_path`) - Method to fetch random file hash from file cache database specified in `db_path`. This method returns dictionary with a key `hash_id` that has value of retrieved random file hash in `str` format.
* __`get_img_by_hash`__(_`db_path`_, _`hash_id`_) - Method to fetch image file data from file cache database specified in `db_path` with specific hash `hash_id`. This method returns a dictionary with the following keys and its values: `hash_id` - image file hash, `file_name` - image file name, `path` - image file path, `type` - image file type, `is_archive` - will be `True` if the file is ZIP archive, `n_content` - for ZIP archive, indicate the number of contained image file,  `container_archive` - the name of containing ZIP archive if the image file is contained in ZIP archive.
* __`get_random_image`__(_`db_path`_) - Method to fetch image file data from file cache database specified in `db_path`. The return value is a dictionary with the same structure as a return value of `get_img_by_hash()` method above.
* __`clear_cache`__(_`db_path`_, `delete_cache = False`) - Method to clear all data from the table in file cache database specified in `db_path`. If the `delete_cache` argument is set to `True`, the file cache SQLite database file will be removed.

## Usage Example

```python

from cache_file import FileCache

# Create file cache object
fileCache = FileCache()

# Create file cache database. Default to './output/file_cache.sqlite')
fileCache.create_file_cache()

# Adding image files contained in a folder or ZIP archive to file cache database
fileCache.add_folder_to_file_cache('./dataset/testdata1.zip')

# Get random file hash_id
hash_dict = fileCache.get_random_hash('./output/file_cache.sqlite')
hash_id = hash_dict['hash_id']

# Fetch image file data from file cache database with specific hash
img_dict = fileCache.get_img_by_hash('./output/file_cache.sqlite', hash_id)
''' Example img_dict:
{'file_name': '10_1.jpg',
 'file_path': './dataset/testdata1.zip/testdata1/10_1.jpg',
 'hash_id': '7bd45969bb6ffc6486fd560e42ab6ed1b788f3bb8541480be893da6ca4fcbff55b7d97e3505dc715fa962a23d24b7325a044206078390c08d63ac03d9fd4f67a',
 'file_type': '.jpg',
 'is_archive': None,
 'n_content': None,
 'container_archive': './dataset/testdata1.zip'
 }
'''
# Fetch random image file data from file cache database.
img_dict = fileCache.get_random_image('./output/file_cache.sqlite')

# Clear all data from the table in file cache database.
fileCache.clear_cache('./output/file_cache.sqlite', delete_cache=False)

```

# CLIP Cache Module
> A tool to create, add and fetch CLIP vector data to .sqlite database file. CLIP cache contains CLIP vector data for each image file in given directory or folder. The attributes for each image file are represented by the following fields: `hash_id`, `clip_vector` and `model`.

## Module Description
The CLIP cache module defined in `cache_clip.py` contains the class definition with the following functions:

* _class_  `cache_clip`.__`ClipCache`__ - A class to construct CLIP cache object.
* __`create_clip_cache`__(_`out_dir = './output'`_, _`db_name = 'clip_cache.sqlite'`_) - Method to create CLIP cache database with default name `clip_cache.sqlite` and default location in project `./output` directory. The CLIP cache database will not be created if the database with same name path already exist.
* __`add_folder_to_clip_cache`__(_`data_dir`_, _`out_dir = './output'`_, _`db_name = 'clip_cache.sqlite'`_) - Method to add image files contained in a folder / directory specified in `data_dir` to CLIP cache database specified in `db_name` at location `out_dir` directory.
* __`get_random_hash`__(_`db_path`) - Method to fetch random file hash from CLIP cache database specified in `db_path`. This method returns dictionary with a key `hash_id` that has value of retrieved random file hash in `str` format.
* __`get_clip_by_hash`__(_`db_path`_, _`hash_id`_) - Method to fetch image CLIP vector from CLIP cache database specified in `db_path` with specific hash `hash_id`. This method returns a dictionary with the following keys and its values: `hash_id` - image file hash in `str` format, `clip_vector` - image CLIP vector in numpy array format, `model` - model type that is used to generate image CLIP vector in `str` format.
* __`get_random_clip`__(_`db_path`_) - Method to fetch image CLIP vector from CLIP cache database specified in `db_path`. The return value is a dictionary with the same structure as a return value of `get_clip_by_hash()` method above.
* __`clear_cache`__(_`db_path`_, `delete_cache = False`) - Method to clear all data from the table in CLIP cache database specified in `db_path`. If the `delete_cache` argument is set to `True`, the CLIP cache SQLite database file will be removed.

## Usage Example

```python

from cache_clip import ClipCache

# Create CLIP cache object
clipCache = ClipCache()

# Create CLIP cache database. Default to './output/clip_cache.sqlite')
clipCache.create_clip_cache()

# Adding image files contained in a folder or ZIP archive to CLIP cache database
clipCache.add_folder_to_clip_cache('./dataset/testdata1.zip')

# Get random file hash_id
hash_dict = clipCache.get_random_hash('./output/clip_cache.sqlite')
hash_id = hash_dict['hash_id']

# Fetch CLIP vector data from CLIP cache database with specific hash
clip_dict = clipCache.get_clip_by_hash('./output/clip_cache.sqlite', hash_id)

# Fetch random CLIP vector data from CLIP cache database.
clip_dict = clipCache.get_random_clip('./output/clip_cache.sqlite')

# Getting data
clip_vector = clip_dict['clip_vector']
hash_id = clip_dict['hash_id']
model = clip_dict['model']

# Clear all data from the table in CLIP cache database.
clipCache.clear_cache('./output/clip_cache.sqlite', delete_cache=False)

```

# Tag Cache Module
> A tool to create, add and fetch image tag data to .sqlite database file. Tag cache contains tag data for each image file in given directory or folder. The attributes for each image file are represented by the following fields: `hash_id` and `tag`.

## Module Description
The tag cache module defined in `cache_tag.py` contains the class definition with the following functions:

* _class_  `cache_tag`.__`TagCache`__ - A class to construct tag cache object.
* __`create_tag_cache`__(_`out_dir = './output'`_, _`db_name = 'tag_cache.sqlite'`_) - Method to create tag cache database with default name `tag_cache.sqlite` and default location in project `./output` directory. The tag cache database will not be created if the database with same name path already exist.
* __`add_folder_to_tag_cache`__(_`data_dir`_, _`out_dir = './output'`_, _`db_name = 'tag_cache.sqlite'`_) - Method to add image files contained in a folder / directory specified in `data_dir` to tag cache database specified in `db_name` at location `out_dir` directory.
* __`get_random_hash`__(_`db_path`) - Method to fetch random file hash from tag cache database specified in `db_path`. This method returns dictionary with a key `hash_id` that has value of retrieved random file hash in `str` format.
* __`get_tag_by_hash`__(_`db_path`_, _`hash_id`_) - Method to fetch image tag from tag cache database specified in `db_path` with specific hash `hash_id`. This method returns a dictionary with the following keys and its values: `hash_id` - image file hash in `str` format, `tag` - image tag in `str` format.
* __`get_random_tag`__(_`db_path`_) - Method to fetch image tag from tag cache database specified in `db_path`. The return value is a dictionary with the same structure as a return value of `get_tag_by_hash()` method above.
* __`clear_cache`__(_`db_path`_, `delete_cache = False`) - Method to clear all data from the table in tag cache database specified in `db_path`. If the `delete_cache` argument is set to `True`, the tag cache SQLite database file will be removed.

## Usage Example

```python

from cache_tag import TagCache

# Create tag cache object
tagCache = TagCache()

# Create tag cache database. Default to './output/tag_cache.sqlite')
tagCache.create_tag_cache()

# Adding image files contained in a folder or ZIP archive to tag cache database
tagCache.add_folder_to_tag_cache('./dataset/testdata1.zip')

# Get random file hash_id
hash_dict = tagCache.get_random_hash('./output/tag_cache.sqlite')
hash_id = hash_dict['hash_id']

# Fetch tag from tag cache database with specific hash
tag_dict = tagCache.get_tag_by_hash('./output/tag_cache.sqlite', hash_id)

# Fetch random tag from tag cache database.
tag_dict = tagCache.get_random_tag('./output/tag_cache.sqlite')

# Getting data
hash_id = tag_dict['hash_id']
tag = tag_dict['tag']
print (f'hash: {hash_id}, tag: {tag}')

# Clear all data from the table in tag cache database.
tagCache.clear_cache('./output/tag_cache.sqlite', delete_cache=False)


```

# Cache Web API Module
> A web API for getting random image file and show its relevant data as follows: file name, file path, image size, container archive, hash ID, request time and the image itself.

## Module Description
The web API module runs FLASK-based server and contains functions to fetch random file from file cache database and returns its respective data (file name, file path, image size, container archive, hash ID, request time and the image itself) based on HTTP request made from web browser.

## Usage Example

Start the web API module form CLI. In default, the server runs on host `0.0.0.0` and port `8080`.
```
python cache_web_api.py
```
or start the web API in other host and port using `host` and `port` CLI arguments as the follows.
```
python cache_web_api.py --host=0.0.0.0 --port=8000
```

Fetch random image from file cache database specified in `db_path`. The following URL request (made from web browser) will return HTML page containing file name, file path, image size, container archive, hash ID, request time and the image itself. Specify the `db_path` as argument with query string using '?' and its value after `=`.

```
http://127.0.0.1:8080/get_random_img?db_path=./output/file_cache.sqlite
```

# Examples

## Listing all the models.
```python 
list_models('./output/models') # listing all the models we have for classification. 
```

## Binary classification of a single image using a single model.

```python
folder_path    = "./images/example1.png" # Input image.
output_dir     = "./classification_single_image_single_model" # outut directory for the classification.
json_file_path = "./input-metadata.json" # metadata .json file path.
bins_number    = 10 # number of bins.
model_path     = "./output/models/model-ovr-logistic-regression-tag-not-pixel-art-digital.pkl" # pickle file for the model.

# Run the classification.
classify_main(
        folder_path    = folder_path , 
        output_dir     = output_dir, 
        json_file_path = json_file_path, 
        bins_number    = bins_number, 
        model_path     = model_path, 
        )

```


## Multiple binary classification for a single image.

```python
folder_path    = "./images/example1.png" # input image.
output_dir     = "./classification_single_image_all_models" # output directory of the classification.
json_file_path = "./input-metadata.json" # metadata .json file path.
bins_number    = 10 # number of bins.
model_path = "./image-tagging-tools/output/models" # pickle file for the model.

# run the classification.
classify_main(
        folder_path    = folder_path , 
        output_dir     = output_dir, 
        json_file_path = json_file_path, 
        bins_number    = bins_number, 
        model_path     = model_path, 
        )
```

## Custom binary classification.

```python

TAG_NAME   = 'not-pixel-art' # tag which you want to classify.
MODEL_TYPE = 'ovr-logistic-regression' # model type you want to use.

folder_path    = "./images/example1.png" # input image.
output_dir     = "./classification_single_image_custom_model" # output directory for the classification.
json_file_path = "./input-metadata.json" # .json fil for metadata.
bins_number    = 10 # bins number 

# generating the path of the model's .pkl file using model type and tag name. 
model_path = generate_model_path(
                                  './output/models',
                                  model_type= MODEL_TYPE,
                                  tag_name= TAG_NAME
                                 )
                                 
classify_main(
        folder_path    = folder_path, 
        output_dir     = output_dir, 
        json_file_path = json_file_path, 
        bins_number    = bins_number, 
        model_path     = model_path, 
        )                                 

```

## Get List of File Hash_ID from Tag Cache Based on List of Models or Specific Model

```python

import json
from cache_tag import TagCache
from api_model import ModelApi

# Specify path to tag cache file
tag_cache_path = './output/tag_cache.sqlite'
# Specify path to directory containing model pickle files or specific model pickle file
models_path = './output/models/'

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
    
    # Output
    print(json.dumps(model_tag_cache_pair, indent=2))

except Exception as e:
    print (f'[ERROR] {e}: Getting data from tag cache failed')

```

## Get List of File Hash_ID from Tag Cache Based on List of Models or Specific Model (CLI Version)

```
python get_tag_cache_by_model.py --tag_cache_path=./output/tag_cache.sqlite --models_path=./output/models/
```

## CLI Arguments

* `tag_cache_path` _[string]_ - _[required]_ - Path to the tag cache file.
* `models_path` _[string]_ - _[required]_ - Path to directory containing model pickle files or specific model pickle file.




