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
python src/to/dir/train.py --metadata_json  '/src/to/dir/pixel-art-tagged-metadata.json' --tag_to_hash_json '/src/to/dir/pixel-art-tagged-tag-to-image-hash-list.json'
```

> Note that if the `output` is not created the script automatically creates it for you. 


> Note that if the `test_per` is not created the script will make test set ~= 10% of the dataset.

Also you may call `--help` to see the options and their defaults in the cli. 

