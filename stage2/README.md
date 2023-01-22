
# Train Classifier
> A script for training classification models based on `input-tag-to-image-hash-list.json` file and `input-metadata.json` file.

## Tool Description
Given a `metadata` json file containing embeddings for images and `tag-to-image-hash` json file containing images' hash with tags, the script start to make for every tag two binary classification models and save it in output folder.

## Installation
All that's needed to start using train.py is to install the dependencies using the command
```
pip install -r ./stage2/requirements.txt
```

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



