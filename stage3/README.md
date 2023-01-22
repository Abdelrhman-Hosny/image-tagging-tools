# Classify Data

> A script for classification models inference given images' `directory` and `metadata_json` .json file. This stage does not process .zip (archived) file.

## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder, the script start to loop over every image and make the classification for it using every binary classification model. If the `--output` argument is not specified, the classification / inference result will be placed at `./output/tagging_output` folder. Time stamp will be appended to folder name (for example: `./output/tagging_output_2023_1_21_0_56`).
In addition, the SQLite database named `score_cache.sqlite` with table named `score_cache` containing file name, file path, file hash, model name, model type, model train date, tag string and tag score for given images will be created in the `./output` folder. 

## Installation
All that's needed to start using classify.py is to install the dependencies using the command
```
pip install -r ./stage3/requirements.txt
```

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


# Classify_single.py
> A script for classification models inference given images path' `image_path`  and models path `model_path`

## Tool Description

Given `image_path` image file path and `model_path` model file path the script classifies single image using specified binary classification model, and saves the result in `single_image_classification` folder 

## Installation
All that's needed to start using classify_single.py is to install the dependencies using the command
```
pip install -r ./stage3/requirements.txt
```

## Example Usage
```
python ./stage3/classify_single.py --image_path=/path/to/img --model_path=path/to/model_file.pkl 
```

## CLI Arguments
* `image_path` _[string]_ - _[required]_ - The path to the image file.
* `model_path` _[string]_ - _[required]_ - The path to the single .pkl file model.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.

Also you may call `--help` to see the options and their defaults in the cli. 
