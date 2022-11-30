

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



