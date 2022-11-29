[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/image-tagging-tools/blob/main/jupyter-notebook-example.ipynb)


# train.py
> A script for training classification models based on `pixel-art-tagged-tag-to-image-hash-list.json` file and `pixel-art-tagged-metadata.json` file.

## Tool Description

Given a `metadata` json file containing embeddings for images and `tag-to-image-hash` json file containing images' hash with tags, the script start to make for every tag two binary classification models and save it in output folder.

## Installation
All that's needed to start using train.py is to install the dependencies using the command
```
pip install -r src/to/dir/requirements.txt
```

## CLI Parameters


* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file. 
* `tag_to_hash_json` _[string]_ - _[required]_ - The path to tag-to-hash json file. 

* `output` _[string]_ - _[optional]_ - The path to the output directory.
* `test_per` _[float]_ - _[optional]_ - The percentage of the test images from the dataset, default = 0.1 

## Example Usage

```
python src/to/dir/train.py --metadata_json  '/src/to/dir/input-metadata.json' --tag_to_hash_json '/src/to/dir/input-tag-to-image-hash-list.json'
```

> Note that if the `output` is not created the script automatically creates it for you. 


> Note that if the `test_per` is not created the script will make test set ~= 10% of the dataset.

Also you may call `--help` to see the options and their defaults in the cli. 


# classify.py
> A script for classification models inference given images' `directory` or .zip file and `metadata_json` .json file.

## Tool Description

Given a `metadata_json` json file containing embeddings for images and `directory` of images' folder or .zip file, the script start to loop over every image and make the classification for it using every binary classification model.

## Installation
All that's needed to start using classify.py is to install the dependencies using the command
```
pip install -r src/to/dir/requirements.txt
```


## CLI Parameters

* `directory` _[string]_ - _[required]_ - The path to the images' folder or images' .zip file. 
* `metadata_json` _[string]_ - _[required]_ - The path to the metadata json file for CLIP embeddings. 
* `output` _[string]_ - _[optional]_ - The path to the output directory for the inference results. 
* `model` _[string]_ - _[optional]_ - The path to the models' .pkl files directory or single .pkl file model.
* `output_bins` _[int]_ - _[optional]_ -  The number of bins of the results for each model.

## Example Usage

```
python src/to/dir/classify.py --metadata_json  '/src/to/dir/input-metadata.json' --directory  /src/to/dir/images_directory 
```

```
python src/to/dir/classify.py --metadata_json  '/src/to/dir/input-metadata.json' \
                              --directory  /src/to/dir/images_directory 
                              --output /src/to/output_dir
                              --output_bins 10
                              --model /src/to/models_dir
```



> Note that if the `output` is not created the script automatically creates it for you. 

> Note that if the `model` is not created the script automatically uses models in [outputs/models](outputs/models/)

Also you may call `--help` to see the options and their defaults in the cli. 



