# ImageDatasetProcessor

> A standalone tool for processing dataset of tagged images and calculating its metadata and the CLIP embeddings. 

## Installation
To use this tool, use python 3.9+ then install the requirements as follows.

```sh
pip install -r ./requirements.txt
```

## Tool Description

Process a directory of images (paths to directory of images or an archived dataset) and computes the images metadata along with its CLIP embeddings and writes the result into a JSON file in specified output folder.

## Example Usage

* Process a tagged dataset and save the output into `./output` folder. In addition, the SQLite database named `dataset_cache.sqlite` with table named `dataset_cache` containing file name, hash and file path for dataset images will be created in the `./output` folder. 

```sh
python ./stage1/ImageDatasetProcessor.py --input_folder=./dataset 
```

* Process a non-tagged dataset and save the output into `./output/clip-cache` folder.  In addition, the SQLite database named `dataset_cache.sqlite` with table named `dataset_cache` containing file name, hash and file path for dataset images will be created in the `./output/clip-cache` folder.

```sh
python ./stage1/ImageDatasetProcessor.py --input_folder=./dataset --tagged_dataset=False
```

The tool will immediately starts working, and output any warnings or error into the std while working. 

## CLI Arguments

* `input_folder` _[string]_ - _[required]_ - path to the directory containing sub-folders of each tag.
* `output_folder` _[string]_ - _[optional]_ - path to the directory where to save the files into it.
* `tagged_dataset` _[bool]_ - _[optional]_ - the dataset to process is a tagged dataset such that each each parent folder name is the tag of the images contained within it, default is `True`

* `clip_model` _[str]_ - _[optional]_ CLIP model to be used, default is `'ViT-B-32'`

* `pretrained` _[str]_ - _[optional]_ -  the pre-trained model to be used for CLIP, default is `'openai'`

* `batch_size` _[int]_ - _[optional]_ -  number of images to process at a time, default is `32`. 
* `num_threads` _[int]_ - _[optional]_ - the number to be used in this process, default is `4`

* `device` _[str]_ - _[optional]_ -  the device to be used in computing the CLIP embeddings, if `None` is provided then `cuda` will be used if available, default is `None`