import sys

from stage1.ImageDatasetProcessor import ImageDatasetProcessor
from stage2.train import main as train_main
from stage4.classify_zip import main as classify_main_zip, zip_gen
from stage4.classify_zip_helper_functions import *

tagged_dataset_path = '/Volumes/docker/mega/KCG/dataset/pixel-art-tagged-v3.zip'
dataset_path = '/Volumes/docker/processing_folder/test-tagging/untagged'
output_folder = '/Volumes/docker/processing_folder/test-tagging/output'
output_models_path = '/Volumes/docker/processing_folder/test-tagging/output/'
metadata_json_path = '/Volumes/docker/processing_folder/test-tagging/output/input-metadata.json' 
tag_to_hash_json = '/Volumes/docker/processing_folder/test-tagging/output/input-tag-to-image-hash-list.json'
tag_model_path     = '/Volumes/docker/processing_folder/test-tagging/output/models/model-ovr-svm-tag-pos-video-game-side-scrolling.pkl'

# Stage 1: Preprocess dataset
# Specify the path to the dataset in tagged_dataset_path variable
tagged_dataset = True
clip_model = 'ViT-B-32'
pretrained = 'openai'
batch_size = 32
num_threads = 4
device = None

ImageDatasetProcessor.process_dataset(
    tagged_dataset_path, 
    output_folder,
    tagged_dataset, 
    clip_model, 
    pretrained,
    batch_size, 
    num_threads, 
    device
)

# Stage 2: Train Classifiers
# Run from ./image-tagging-tools directory
metadata_json = metadata_json_path
output_dir = output_folder
test_per = 0.1

train_main(
    metadata_json = metadata_json,
    tag_to_hash_json = tag_to_hash_json,
    output_dir = output_dir,
    test_per = test_per
)

list_models(output_models_path) # listing all the models we have for classification. 

# Stage 3: Classify data for non zip folder
# data_path    = '../path/to/image/data/folder/'
# output_dir     = './output/classification_all_images_all_models'
# json_file_path = './output/input-metadata.json'
# bins_number    = 10
# # Specify path to folder containing all models in model_path variable
# model_path     = output_models_path
# classify_main(
#         folder_path    = data_path, 
#         output_dir     = output_dir, 
#         json_file_path = json_file_path, 
#         bins_number    = bins_number, 
#         model_path     = model_path, 
#         )

# Stage 4: Classify data for zip folder
data_path    = dataset_path
output_dir     = output_folder
json_file_path = metadata_json_path
bins_number    = 10
# Specify path to folder containing all models in model_path variable
model_path     = output_models_path

classify_main_zip(
        folder_path    = data_path, 
        output_dir     = output_dir, 
        json_file_path = json_file_path, 
        bins_number    = bins_number, 
        model_path     = model_path, 
        )

# Get single tag score
folder_path    = dataset_path
clip_model , preprocess , device = get_clip(clip_model_type= 'ViT-B-32',pretrained= 'openai')
model_dict = create_models_dict(tag_model_path)
print (model_dict)

# Loop through each zip file.
for file in [folder_path]:
    # Generating images
    for img, img_file_name in zip_gen(file):
        # Calculate score
        score = get_single_tag_score(img, img_file_name, model_dict, clip_model, preprocess, device)
        print (f'[INFO] Score: {score}')

print("[INFO] Finished.")