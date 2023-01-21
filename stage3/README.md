# classify.py
> A script for classification models inference given images' `directory` and `metadata_json` .json file. This stage does not process .zip (archived) file.

## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder, the script start to loop over every image and make the classification for it using every binary classification model. This stage does not process .zip (archived) file. 
In addition, the SQLite database named `score_cache.sqlite` (containing file name, file path, hash, model type, tag name and tag score for given images) will be created in the `output` folder in the root directory. 

## Installation
All that's needed to start using classify.py is to install the dependencies using the command
```
pip install -r src/to/dir/requirements.txt
```


## CLI Parameters

* `directory` _[string]_ - _[required]_ - The path to the images folder (not in .zip format). 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.

## Example Usage

```
python ./stage3/classify.py --metadata_json  './output/input-metadata.json' --directory /src/to/dir/images_directory 
```

> Note that if the `output` is not created the script automatically creates it for you. 

> Note that if the `model` is not created the script automatically uses models in [outputs/models](outputs/models/)

Also you may call `--help` to see the options and their defaults in the cli. 

<br/>
<br/>

# classify_single.py
> A script for classification models inference given images path' `image_path`  and models path `model_path`

## Tool Description

Given a `image_path` image file path and `model_path` model file path the script classifies single image using specified binary classification model, and saves the result in `single_image_classification` folder 

## Installation
All that's needed to start using classify_single.py is to install the dependencies using the command
```
pip install -r src/to/dir/requirements.txt
```


## CLI Parameters

* `image_path` _[string]_ - _[required]_ - The path to the image file.
* `model_path` _[string]_ - _[required]_ - The path to the single .pkl file model.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.

## Example Usage

```
python src/to/dir/classify_single.py --image_path  '/src/to/dir/image.png' --model_path  /src/to/dir/model_file.pkl 
```

Also you may call `--help` to see the options and their defaults in the cli. 
