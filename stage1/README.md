# ImageDatasetProcessor
> A standalone tool for for processing dataset of tagged images and calculating its metadata and the CLIP embeddings. 

## Tool Description

process a directory of images (paths to directory of images or an archived dataset), and computes the images metadata along with its CLIP embeddings and writes the result into a JSON file into `output_folder`

## Installation
All that's needed to start using ImageDatasetProcessor is to install the dependencies using the command
```
pip install -r src/to/dir/requirements.txt
```

## CLI Parameters

* `input_folder` _[string]_ - _[required]_ - path to the directory containing sub-folders of each tag.
* `output_folder` _[string]_ - _[optional]_ - path to the directory where to save the files into it.
* `tagged_dataset` _[bool]_ - _[optional]_ - the dataset to process is a tagged dataset such that each each parent folder name is the tag of the images contained within it, default is `True`


* `clip_model` _[str]_ - _[optional]_ CLIP model to be used, default is `'ViT-B-32'`

* `pretrained` _[str]_ - _[optional]_ -  the pre-trained model to be used for CLIP, default is `'openai'`

* `batch_size` _[int]_ - _[optional]_ -  number of images to process at a time, default is `32`. 
* `num_threads` _[int]_ - _[optional]_ - the number to be used in this process, default is `4`

* `device` _[str]_ - _[optional]_ -  the device to be used in computing the CLIP embeddings, if `None` is provided then `cuda` will be used if available, default is `None`

## Example Usage

```
python src/to/dir/ImageDatasetProcessor.py --input_folder='./my-dataset' output_folder='./result'
```

> Note that if the `output_folder` is not created the tool automatically creates it for you. 

The tool will immediately starts working, and output any warnings or error into the std while working. 
