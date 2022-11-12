# image-tagging-tools
Tools for Tagging and Cleaning Up Image Datasets.

OpenClip model used :: https://github.com/mlfoundations/open_clip

# Classification 
> `classify.py` : a script for classifying images using binary classifiers and output all the results to it's class folder.
                  Loops on all the models.
                  
> `classify_bins.py` : a script for classifying images using binary classifiers and output all the results to it's class bin folder.
                       Loops on all the models.
                  

## Tool Description

Given a `directory` (images directory or zip file) containing images, It classifies the images in this directory and put the result in image_tagging output folder.   

## Installation
```
pip install -r requirements.txt
```

## CLI Parameters


* `directory` _[string]_ - _[required]_ - The source directory or zip file of the dataset containing the images. 

## Example Usage
>  `classify.py` Usage:

```
python classify.py --directory path/to/directory-or-zip-file
```

>  `classify_bins.py` Usage:

```
python classify_bins.py --directory path/to/directory-or-zip-file
```



