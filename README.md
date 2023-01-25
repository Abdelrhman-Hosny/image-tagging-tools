

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
python ./stage2/train.py --metadata_json=./output/input-metadata.json --tag_to_hash_json=./output/input-tag-to-image-hash-list.json
```

> If the `output` folder is not present, the script automatically creates it for you. 
Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Arguments

* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file. 
* `tag_to_hash_json` _[string]_ - _[required]_ - The path to tag-to-hash json file. 
* `output` _[string]_ - _[optional]_ - The path to the output directory.
* `test_per` _[float]_ - _[optional]_ - The percentage of the test images from the dataset, default = 0.1. If the `test_per` is not created the script will make test set ~= 10% of the dataset.

# Stage 3: Classify Data

> A script for classification models inference given images' `directory` and `metadata_json` .json file. This stage does not process .zip (archived) file.

## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder, the script start to loop over every image and make the classification for it using every binary classification model. If the `--output` argument is not specified, the classification / inference result will be placed at `./output/tagging_output` folder. Time stamp will be appended to folder name (for example: `./output/tagging_output_2023_1_21_0_56`).
In addition, the SQLite database named `score_cache.sqlite` with table named `score_cache` containing file name, file path, file hash, model name, model type, model train date, tag string and tag score for given images will be created in the `./output` folder. 


## Example Usage

```
python ./stage3/classify.py --metadata_json=./output/input-metadata.json --directory=/path/to/images/dir
```
Or

```
python ./stage3/classify.py --metadata_json=./output/input-metadata.json --directory=./path/to/images/dir --output=./output --output_bins=10 --model=./output/models

```

> Note that if the `output` folder is not present, the script automatically creates it for you. 
Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Arguments

* `directory` _[string]_ - _[required]_ - The path to the test images folder. 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model. If this is not specified, the script automatically use models in `./output/models` directory.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.


# Stage 4: Classify Data (for ZIP Archives)

> A script for classification models inference given images' `directory` and `metadata_json` .json file. This stage only process .zip (archived) file. It is used if the image data specified in `directory` argument is either in .zip archived format or containing images archived in .zip format. 

## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder, the script start to loop over every image and make the classification for it using every binary classification model. If the `--output` argument is not specified, the classification / inference result will be placed at `./output/tagging_output` folder. Time stamp will be appended to folder name (for example: `./output/tagging_output_2023_1_21_0_56`).

In addition, the SQLite database named `zip_score_cache.sqlite` with table named `zip_score_cache` containing file name, file path, archive path, type of file, hash, model type, tag name and tag score for given images will be created in the `output` folder. 

## Example Usage

```
python ./stage4/classify_zip.py --metadata_json=./output/input-metadata.json --directory=/path/to/images/dir
```
Or
```
python ./stage4/classify_zip.py --metadata_json=./output/input-metadata.json --directory=./src/to/images/dir --output=./output --output_bins=10 --model=./output/models
```

> Note that if the `output` folder is not present, the script automatically creates it for you. 
Also you may call `--help` to see the options and their defaults in the cli. 

## CLI Arguments

* `directory` _[string]_ - _[required]_ - The path to the images folder or images .zip file. 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model. If this is not specified, the script automatically use models in `./output/models` directory.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.


# Dataset File Cache Module
> A tool to create file cache in the form of SQLite database, add data to and fetch from it. File cache contains attributes for each file in given directory or folder. The attributes for each file are represented by the following fields in file cache SQLite database table: `hash_id`, `file_name`, `path`, `type`, `is_archive`, `n_content` and `container_archive`.

## Module Description
The file cache module defined in `cache_file.py` contains the class definition with the following functions:

* _class_  `cache_file`.__`FileCache`__ - A class to construct file cache object.
* __`create_file_cache`__(_`out_dir = './output'`_, _`db_name = 'file_cache.sqlite'`_) - Method to create file cache database with default name `file_cache.sqlite` and default location in project `./output` directory. The file cache database will not be created if the database with same name path already exist.
* __`add_folder_to_file_cache`__(_`data_dir`_, _`out_dir = './output'`_, _`db_name = 'file_cache.sqlite'`_) - Method to add image files contained in a folder / directory specified in `data_dir` to file cache database specified in `db_name` at location `out_dir` directory.
* __`get_random_hash`__(_`db_path`) - Method to fetch random file hash from file cache database specified in `db_path`. This method returns random file hash in `str` format.
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
hash_id = fileCache.get_random_hash('./output/file_cache.sqlite')

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

# Dataset CLIP Cache Module
> A tool to create, add and fetch CLIP vector data to .sqlite database file. CLIP cache contains CLIP vector data for each image file in given directory or folder. The attributes for each image file are represented by the following fields: `hash_id`, `clip_vector` and `model`.

## Module Description
The CLIP cache module defined in `cache_clip.py` contains the class definition with the following functions:

* _class_  `cache_clip`.__`ClipCache`__ - A class to construct CLIP cache object.
* __`create_clip_cache`__(_`out_dir = './output'`_, _`db_name = 'clip_cache.sqlite'`_) - Method to create CLIP cache database with default name `clip_cache.sqlite` and default location in project `./output` directory. The CLIP cache database will not be created if the database with same name path already exist.
* __`add_folder_to_clip_cache`__(_`data_dir`_, _`out_dir = './output'`_, _`db_name = 'clip_cache.sqlite'`_) - Method to add image files contained in a folder / directory specified in `data_dir` to CLIP cache database specified in `db_name` at location `out_dir` directory.
* __`get_random_hash`__(_`db_path`) - Method to fetch random file hash from CLIP cache database specified in `db_path`. This method returns random file hash in `str` format.
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
hash_id = clipCache.get_random_hash('./output/clip_cache.sqlite')

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

# Dataset Tag Cache Module
> A tool to create, add and fetch image tag data to .sqlite database file. Tag cache contains tag data for each image file in given directory or folder. The attributes for each image file are represented by the following fields: `hash_id` and `tag`.

## Module Description
The tag cache module defined in `cache_tag.py` contains the class definition with the following functions:

* _class_  `cache_tag`.__`TagCache`__ - A class to construct tag cache object.
* __`create_tag_cache`__(_`out_dir = './output'`_, _`db_name = 'tag_cache.sqlite'`_) - Method to create tag cache database with default name `tag_cache.sqlite` and default location in project `./output` directory. The tag cache database will not be created if the database with same name path already exist.
* __`add_folder_to_tag_cache`__(_`data_dir`_, _`out_dir = './output'`_, _`db_name = 'tag_cache.sqlite'`_) - Method to add image files contained in a folder / directory specified in `data_dir` to tag cache database specified in `db_name` at location `out_dir` directory.
* __`get_random_hash`__(_`db_path`) - Method to fetch random file hash id from tag cache database specified in `db_path`. This method returns random file hash in `str` format.
* __`get_tag_by_hash`__(_`db_path`_, _`hash_id`_) - Method to fetch image tag from tag cache database specified in `db_path` with specific hash `hash_id`. This method returns image tag in `str` format.
* __`get_hash_by_tag`__(_`db_path`_, _`tag`_) - Method to get list of hash ids from tag cache database specified in `db_path` for given tag string `tag`. 
* __`get_random_tag`__(_`db_path`_) - Method to fetch image tag from tag cache database specified in `db_path`. This method returns a dictionary with the following keys and its values: `hash_id` - image file hash id in `str` format, `tag` - image tag in `str` format.
The return value is a dictionary with the same structure as a return value of `get_tag_by_hash()`
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
hash_id = tagCache.get_random_hash('./output/tag_cache.sqlite')

# Fetch tag string from tag cache database with specific hash
tag_str = tagCache.get_tag_by_hash('./output/tag_cache.sqlite', hash_id)

# Fetch list of hash ids for specific tag string
hash_id_list = tagCache.get_hash_by_tag('./output/tag_cache.sqlite', tag_str)

# Fetch random hash_id and tag pair from tag cache database.
tag_dict = tagCache.get_random_tag('./output/tag_cache.sqlite')
hash_id = tag_dict['hash_id']
tag = tag_dict['tag']
print (f'hash: {hash_id}, tag: {tag}')
'''Example output
{'hash_id': '5fed6cf0ff1028f58f6cc73bb09e142eb4aacf1f5a206327fda3ef3db8cfcbf3e643ec1126238115d0fcfbe35be6e3204b21e06d1c3a7f0229e6aa18696ee5da', 'tag': 'other-validation'}
'''

# Clear all data from the table in tag cache database.
tagCache.clear_cache('./output/tag_cache.sqlite', delete_cache=False)

```

# File Cache Web API Module
> A web API for getting random image file and show its relevant data as follows: file name, file path, image size, container archive, hash ID, request time and the image itself.

## Module Description
The web API module defined in `file_cache_web_api.py` runs FLASK-based server and contains functions to fetch random file from file cache database and returns its respective data (file name, file path, image size, container archive, hash ID, request time and the image itself) based on HTTP request made from web browser.

## Usage Example

Start the web API module form CLI. In default, the server runs on host `0.0.0.0` and port `8080`.
```
python file_cache_web_api.py
```
or start the web API in other host and port using `host` and `port` CLI arguments as the follows.
```
python file_cache_web_api.py --host=0.0.0.0 --port=8000
```

Fetch random image from file cache database (created using File Cache Module in `file_cache.py`) specified in `db_path`. The following URL request (made from web browser) will return HTML page containing file name, file path, image size, container archive, hash ID, request time and the image itself. Specify the `db_path` as argument with query string using '?' and its value after `=`.

```
http://127.0.0.1:8080/get_random_img?db_path=./output/file_cache.sqlite
```

# Model API
> An API to list, access and use existing classifier models. Model API contains function that accesses existing classifier model pickle files (in given path) and returns existing classifier model as Python dictionary.

## Module Description
The model API defined in `api_model.py` contains the class definition with the following functions:

* _class_  `api_model`.__`ModelApi`__ - A class to construct the model loader object.
* __`get_models_dict`__(_`models_path`_) - Method that returns models dictionary for model pickle file in given `models_path`.

## Usage Example
```python

from api_model import ModelApi

# Create model loader object
model_api = ModelApi()

# Get models dictionary to models_dict for model pickle files in given models_path.
models_dict = model_api.get_models_dict(models_path='./output/models')
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
```

# Model Web API
> Web API to list existing classifier models (name, type, training start time and tag string) in JSON format. 

## Module Description
The model web API defined in `webapi_model.py` runs FLASK-based server and contains functions to list existing classifier models (name, type, training start time and tag string) in JSON format based on HTTP request made from web browser.

## Usage Example
Start the model web API form CLI. In default, the server runs on host `0.0.0.0` and port `8080`.
```
python webapi_model.py
```
or start the web API in other host and port using `host` and `port` CLI arguments as the follows.
```
python webapi_model.py --host=0.0.0.0 --port=8000
```

List existing classifier models (name, type, training start time and tag string)

```
http://127.0.0.1:8080/get_models
```

# Model Cache
> A tool for:
* Get list of image files for specific model and score range from classification result / score cache created in classification Stage 3 and Stage 4 (`score_cache.sqlite` or `zip_score_cache.sqlite`)
* Clearing classification result / score cache created in classification Stage 3 and Stage 4 (`score_cache.sqlite` or `zip_score_cache.sqlite`) from entry with model training date older than the training date of respective current models in specified models directory.

## Module Description
The model cache defined in `model_cache.py` contains the class definition with the following functions:

* _class_  `model_cache`.__`ModelCache`__ - A class to construct the model cache object.
* __`get_img_from_score_cache`__(_`models_name`_, _`score_gte = 0.0`_, _`score_lte = 1.0`_, _`db_path = './output/score_cache.sqlite'`_, _`db_table_name = 'score_cache'`_) - Returns list of file names for specific `model_name` and score between `score_gte` and `score_lte` from classification cache in `db_path` with table name `db_table_name`.
* __`clear_score_cache_by_model_date`__(_`models_path = './output/models'`_, _`score_cache_path = './output/score_cache.sqlite'`_, _`score_cache_table_name = 'score_cache'`_) - Clearing classification result / score cache in `score_cache_path` from entry with model training date older than the training date of respective current models (in `models_path`).

## Usage Example

Get list of image files for specific model and score range

```python

from model_cache import ModelCache

# Create model cache object
model_cache = ModelCache()

# Get files dictionary 
files_dict = model_cache.get_img_from_score_cache(model_name='model-ovr-logistic-regression-tag-pos-character', score_gte=0.9, score_lte=1.0)

```

Clearing score cache based on model's training date. The following example will eliminate entry in score cache `score_cache.sqlite` if the model's training date is older than the training date of respsctive model in `./output/models` directory.


```python

from model_cache import ModelCache

# Create model cache object
model_cache = ModelCache()

# Clearing score cache for entries with outdated model 
is_success, deleted_entries = model_cache.clear_score_cache_by_model_date()
# is_success will be True when success or False if there is an error
# deleted_entries will contain list of deleted entry dictionary.
# The method will print the original number of entries, number of deleted entriee and current number of entries.

```

Get list of image files for specific model and score range (Web API version)

## Usage Example
Start the model cache web API form CLI. In default, the server runs on host `0.0.0.0` and port `8080`.
```
python webapi_model_cache.py
```
or start the web API in other host and port using `host` and `port` CLI arguments as the follows.
```
python webapi_model_cache.py --host=0.0.0.0 --port=8000
```

List existing classifier models (name, type, training start time and tag string)
```
http://127.0.0.1:8080/get_images_in_tag_score_range?model_name=model-ovr-logistic-regression-tag-not-pixel-art&score_gte=0.9&score_lte=1.0
```

