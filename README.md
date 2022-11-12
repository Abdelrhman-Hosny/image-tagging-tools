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
git clone https://github.com/kk-digital/image-tagging-tools.git
cd image-tagging-tools
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

# Classification Using Single Model 

> `specific_classifier_bins.py` : a script for classifying images using binary classifier and output all the results to it's class folder with bins sub-folders.
                  
                           
## Tool Description

Given a `directory` (images directory or zip file) containing images, It classifies the images in this directory and put the result in image_tagging output folder.   
Given a `model` (pickle file of a model), The path of the pickle file used to load the model for classifiction. see :    

## Installation
```
git clone https://github.com/kk-digital/image-tagging-tools.git
cd image-tagging-tools
pip install -r requirements.txt
```

## CLI Parameters


* `directory` _[string]_ - _[required]_ - The source directory or zip file of the dataset containing the images. 
* `model` _[string]_ - _[required]_ - The source pickle file for the model. see: [models](./outputs/models)

## Example Usage
>  `specific_classifier_bins.py` Usage:

```
python specific_classifier_bins.py --directory path/to/directory-or-zip-file  --model /path/to/model_pickle_file.pkl
```





