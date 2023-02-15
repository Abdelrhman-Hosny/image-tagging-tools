import sys
import os
from tqdm import tqdm 
from stage1 import ImageDatasetProcessor
from stage2.train import main as train_main
from stage4.classify_zip import main as classify_main_zip, zip_gen, list_models, get_clip, create_models_dict, get_single_tag_score

tagged_dataset_path = '/Volumes/docker/mega/KCG/dataset/pixel-art-tagged-v3.zip'
dataset_path = '/Volumes/docker/processing_folder/test-tagging/untagged'
output_folder = '/Volumes/docker/processing_folder/test-tagging/output'

output_models_path = output_folder
metadata_json_path = os.path.join(output_folder, 'input-metadata.json')
tag_to_hash_json = os.path.join(output_folder, 'input-tag-to-image-hash-list.json')

# Stage 1: Preprocess dataset
# Specify the path to the dataset in tagged_dataset_path variable
tagged_dataset = True
clip_model = 'ViT-B-32'
pretrained = 'openai'
batch_size = 32
num_threads = 4
device = None

ImageDatasetProcessor.ImageDatasetProcessor.process_dataset(
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