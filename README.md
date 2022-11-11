# image-tagging-tools
Tools for Tagging and Cleaning Up Image Datasets.

OpenClip model used :: https://github.com/mlfoundations/open_clip

# Classification 
> A script for classifying images using binary classifiers and output all the results in a folder.

## Tool Description

Given a `directory` (images directory or zip file) containing images, It classifies the images in this directory and put the result in image_tagging output folder.   

## Installation
```
pip install -r requirements.txt
```

## CLI Parameters


* `directory` _[string]_ - _[required]_ - The source directory or zip file of the dataset containing the images. 

## Example Usage

```
python classify.py --directory path/to/directory-or-zip-file
```


