# Classify Data (for ZIP Archives)

> A script for classification models inference given images' `directory` and `metadata_json` .json file. This stage only process .zip (archived) file. It is used if the image data specified in `directory` argument is either in .zip archived format or containing images archived in .zip format. 

## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder, the script start to loop over every image and make the classification for it using every binary classification model. If the `--output` argument is not specified, the classification / inference result will be placed at `./output/tagging_output` folder. Time stamp will be appended to folder name (for example: `./output/tagging_output_2023_1_21_0_56`).

In addition, the SQLite database named `zip_score_cache.sqlite` with table named `zip_score_cache` containing file name, file path, archive path, type of file, hash, model type, tag name and tag score for given images will be created in the `output` folder. 

## Installation
All that's needed to start using classify_zip.py is to install the dependencies using the command
```
pip install -r ./stage4/requirements.txt
```

## Example Usage

```
python ./stage4/classify_zip.py --metadata_json=./output/input-metadata.json --directory=/path/to/images/dir
```

> Note that if the `output` is not created the script automatically creates it for you. 
Also you may call `--help` to see the options and their defaults in the cli. 


## CLI Arguments

* `directory` _[string]_ - _[required]_ - The path to the images folder or images .zip file. 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model. If this is not specified, the script automatically use models in `./output/models` directory.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.
