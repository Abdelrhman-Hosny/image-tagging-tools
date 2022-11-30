import os
from typing import Tuple, Union
import open_clip
import torch
import numpy as np
from PIL import Image
from ImageDatasetLoader import ImageDatasetLoader
import hashlib
import json 
import fire
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

class ImageDatasetProcessor: 
    """wrapper that contains the utility methods to process a dataset given its path, that computes the metadata of an image 
        dataset along with its CLIP embeddings
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def __compute_blake2b(image: Image.Image) -> str: 
        """compute the BLAKE2b of a PIL image. 
        :param image: The PIL image to compute its BLAKE2b
        :type image: PIL.Image.Image
        :returns: the BLAKE2b str of the given image. 
        :rtype: str
        """
        
        return hashlib.blake2b(image.tobytes()).hexdigest()
    @staticmethod
    def __compute_sha256(image: Image.Image) -> str: 
        """compute the SHA256 of a PIL image. 
        :param image: The PIL image to compute its sha256
        :type image: PIL.Image.Image
        :returns: the sha256 of the given image. 
        :rtype: str
        """
        return hashlib.sha256(image.tobytes()).hexdigest()
    @staticmethod
    def __write_to_json(to_write: dict, path: str) -> None: 
        """method to write python dictionary into json file.
        :param to_write: The dict to be written into JSON
        :type to_write: dict
        :param path: the output path of the JSON file. 
        :type path: str
        :returns: writes the file into the specified path. 
        :rtype: None
        """
        
        with open(path, mode = 'w', encoding = 'utf-8') as json_file: 
            json.dump(to_write, json_file, indent = 4)
        
    @staticmethod
    def __save_dataset_metadata_json(json_result: dict, output_path: str, dataset_name: str) -> None:
        """method save the metadata of a dataset into a given path as `{output_path}/{dataset_name}-metadata.json`. 
        :param json_result: The dict storing the metadata of dataset. 
        :type json_result: dict
        :param output_path: the path of the directory to save the json file at. 
        :type output_path: str
        :param dataset_name: name of the dataset. 
        :type dataset_name: str
        :returns: writes the file into the specified path. 
        :rtype: None
        """
        
        output_path = os.path.join(output_path, "input-metadata.json")
        ImageDatasetProcessor.__write_to_json(json_result, output_path)

    @staticmethod
    def __save_hash_to_tags_list_json(json_result: dict, output_path: str, dataset_name: str) -> None:
        """method to save the tags for each image in a dataset into a given path as `{output_path}/{dataset_name}-image-hash-to-tag-list.json`. 
        :param json_result: The dict storing the metadata of dataset. 
        :type json_result: dict
        :param output_path: the path of the directory to save the json file at. 
        :type output_path: str
        :param dataset_name: name of the dataset. 
        :type dataset_name: str
        :returns: writes the file into the specified path. 
        :rtype: None
        """
        
        output_path = os.path.join(output_path, "input-image-hash-to-tag-list.json")
        hash_to_tag_list = {} 
        
        for image_hash, metadata in json_result.items(): 
            hash_to_tag_list[image_hash] = metadata['image_tags']
            
            
        ImageDatasetProcessor.__write_to_json(hash_to_tag_list, output_path)
    
    @staticmethod
    def __save_tags_to_images_hash_list_json(json_result: dict, output_path: str, dataset_name: str) -> None:
        """method to save the images for each tag of a dataset into a given path as `{output_path}/{dataset_name}-tag-to-image-hash-list.json`. 
        :param json_result: The dict storing the metadata of dataset. 
        :type json_result: dict
        :param output_path: the path of the directory to save the json file at. 
        :type output_path: str
        :param dataset_name: name of the dataset. 
        :type dataset_name: str
        :returns: writes the file into the specified path. 
        :rtype: None
        """
        
        output_path = os.path.join(output_path, "input-tag-to-image-hash-list.json")
        tag_to_hash_list = {} 
        
        for image_hash, metadata in json_result.items():
            image_tags = metadata['image_tags']
            
            for tag in image_tags: 
                if tag not in tag_to_hash_list: 
                    tag_to_hash_list[tag] = []
                
                tag_to_hash_list[tag].append(image_hash)
            
            
        ImageDatasetProcessor.__write_to_json(tag_to_hash_list, output_path)

    @staticmethod
    def __image_metadata(image: Image.Image, image_id: int) -> Tuple[dict, int]: 
        """Given a PIL image this method computes its metadata and returns it as dictionary 
            along with the provided `image_id` (used to know the image from returned multithreading)
            
        :param image: The PIL image to write its metadata. 
        :type image: PIL.Image.Image
        :param image_id: id of the given image (used to specify for multithreading)
        :type image_id: int
        :returns: returns a Tuple with (metadata, image_id)
        :rtype: Tuple[dict, int]
        """
        
        metadata = {}
        
        try:
            
            image_path = os.path.abspath(image.filename)
            tag_name = Path(image_path).parent.name
            
            metadata = {
                'hash_id': ImageDatasetProcessor.__compute_blake2b(image), 
                'file_path': image_path, 
                'file_name': os.path.split(image.filename)[1], 
                'file_type': image.format,
                'image_size_bytes': os.stat(image_path).st_size,
                'image_resolution': image.size,
                'image_xsize': image.size[0], 
                'image_ysize': image.size[1],
                'image_tags': [tag_name], 
            }
        except Exception as error: 
            try: 
                print(f"Failed to compute the metadata for the image {image.filename} because of the error message {error}")
            except Exception: 
                pass 
        
        return (metadata, image_id)

    @staticmethod
    def process_dataset(
        dataset_path: str,
        output_folder: str,
        clip_model: str = 'ViT-B-32',
        pretrained: str = 'openai',
        batch_size: int = 32,
        num_threads: int = 4,
        device: Union[str, None] = None, 
    ) -> None: 
        """process a datasets of images (directory of images or an archived dataset), and computes its metadata 
            along with its CLIP embeddings and writes the result into a JSON file into `output_folder`
            
        :param dataset_path: path of the dataset whatever it's an archive or a directory of images. 
        :type dataset_path: str
        :param output_folder: path to the directory where to save the files into it. 
        :type output_folder: str
        :param clip_model: CLIP model to be used, default is `'ViT-B-32'`
        :type clip_model: str
        :param pretrained: the pre-trained model to be used for CLIP, default is `'openai'`
        :type pretrained: str
        :param batch_size: number of images to process at a time, default is `32`. 
        :type batch_size: int
        :param num_threads: the number to be used in this process, default is `4`
        :type num_threads: int
        :param device: the device to be used in computing the CLIP embeddings, if `None` is provided then `cuda` will be used if available, default is `None`
        :type device: Union[str, None]
        :returns: process the provided dataset and write the result into `output_folder`
        :rtype: None
        """
        
        #make sure output directory is created 
        os.makedirs(output_folder, exist_ok = True)

        #the name of the file/dataset
        dataset_name = os.path.splitext(os.path.split(dataset_path)[1])[0]

        #init the thread pool. 
        thread_pool = ThreadPoolExecutor(max_workers = num_threads)
        
        #detect the device to be used in calculating the embeddings. 
        if device is None: 
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #load the CLIP model. 
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model,pretrained = pretrained)
        
        model = model.to(device)
        images_loader = None
        #load the image dataset. 
        try: 
            images_loader = ImageDatasetLoader.load(dataset_path, recursive = True, batch_size = batch_size)
        except AssertionError as error: 
            print(f"[Error] in dataset: {dataset_name} due to the error: {error}")
    
        processed_images_count = 0 
        json_result = {}

        with torch.no_grad(): 
            
            #loop over the image dataset. 
            for images_chunk in images_loader: 
                #preprocess the images batch 
                preprocessed_chunk = torch.stack([preprocess(image) for image in images_chunk])
                #compute the metadata of the images batch. 
                tasks = [thread_pool.submit(ImageDatasetProcessor.__image_metadata, image, image_index,) for image_index, image in enumerate(images_chunk)]
                #compute the CLIP embeddings of the current batch of images. 
                images_embeddings = model.encode_image(preprocessed_chunk.to(device))
                
                #wait for each thread to finish and match the resulted embeddings with the metadata computed from threads. 
                for task in as_completed(tasks): 
                    metadata, image_index = task.result() 
                    
                    #update the metadata with the image embeddings.
                    metadata.update({
                        'embeddings_type': 'CLIP', 
                        'model': clip_model, 
                        'pretrained': pretrained,
                        'embeddings_vector': images_embeddings[image_index].tolist(), 
                    })
                    
                    #hash of the image (BLAKE2b hash) is the key of the image data.
                    if metadata['hash_id'] in json_result:
                        tag = metadata['image_tags'][0]
                        if tag not in json_result[metadata['hash_id']]['image_tags']: 
                            json_result[metadata['hash_id']]['image_tags'].append(tag)
                    else: 
                        json_result[metadata['hash_id']] = metadata

                    processed_images_count += 1
                    
                    if processed_images_count % 1000 == 0: 
                        print(f"[INFO] dataset {dataset_name} finished processing {processed_images_count} images so far")

        #save the metadata of the dataset
        ImageDatasetProcessor.__save_dataset_metadata_json(json_result, output_folder, dataset_name)
        #save the json file of hash to tags list.
        ImageDatasetProcessor.__save_hash_to_tags_list_json(json_result, output_folder, dataset_name)
        #save the json file of tags to images hash list. 
        ImageDatasetProcessor.__save_tags_to_images_hash_list_json(json_result, output_folder, dataset_name)
                
        thread_pool.shutdown() #make sure all threads were terminated. 

    @staticmethod
    def process(
        input_folder: str,
        output_folder: str,
        clip_model: str = 'ViT-B-32',
        pretrained: str = 'openai',
        batch_size: int = 32,
        num_threads: int = 4,
        device: Union[str, None] = None,
    ) -> None: 
        """process a directory of images (paths to directory of images or an archived dataset), and computes the 
    images metadata along with its CLIP embeddings and writes the result into a JSON file into `output_folder`
            
        :param input_folder: path to the directory containing sub-folders of each tag. 
        :type input_folder: str
        :param output_folder: path to the directory where to save the files into it. 
        :type output_folder: str
        :param clip_model: CLIP model to be used, default is `'ViT-B-32'`
        :type clip_model: str
        :param pretrained: the pre-trained model to be used for CLIP, default is `'openai'`
        :type pretrained: str
        :param batch_size: number of images to process at a time, default is `32`. 
        :type batch_size: int
        :param num_threads: the number to be used in this process, default is `4`
        :type num_threads: int
        :param device: the device to be used in computing the CLIP embeddings, if `None` is provided then `cuda` will be used if available, default is `None`
        :type device: Union[str, None]
        :returns: process the provided dataset and write the result into `output_folder`
        :rtype: None
        """
        
        #make sure result output path exists 
        os.makedirs(output_folder, exist_ok = True)

        ImageDatasetProcessor.process_dataset(
            input_folder,
            output_folder,
            clip_model,
            pretrained,
            batch_size, 
            num_threads,
            device if device is None else device.lower(), 
        )
                
        return 

def process_image_dataset_cli(
    input_folder: str,
    output_folder: str, 
    clip_model: str = 'ViT-B-32',
    pretrained: str = 'openai',
    batch_size: int = 32,
    num_threads: int = 4,
    device: Union[str, None] = None, 
) -> None: 
    """process a directory of images (paths to directory of images or an archived dataset), and computes the 
    images metadata along with its CLIP embeddings and writes the result into a JSON file into `output_folder`
        
    :param input_folder: path to the directory containing sub-folders of each tag. 
    :type input_folder: str
    :param output_folder: path to the directory where to save the files into it. 
    :type output_folder: str
    :param clip_model: CLIP model to be used, default is `'ViT-B-32'`
    :type clip_model: str
    :param pretrained: the pre-trained model to be used for CLIP, default is `'openai'`
    :type pretrained: str
    :param batch_size: number of images to process at a time, default is `32`. 
    :type batch_size: int
    :param num_threads: the number to be used in this process, default is `4`
    :type num_threads: int
    :param device: the device to be used in computing the CLIP embeddings, if `None` is provided then `cuda` will be used if available, default is `None`
    :type device: Union[str, None]
    :returns: process the provided dataset and write the result into `output_folder`
    :rtype: None
    """
    
    # #check if the provided path is for a directory of datasets or a list of datasets paths. 
    # if directory_of_datasets: 
    #     if type(datasets_paths) == str:
    #         datasets_paths = [os.path.join(datasets_paths, path) for path in os.listdir(datasets_paths)]
    #     else: 
    #         raise TypeError("you should provide a single path as a string of directory full of datasets")
    
    #check if the provided device are one of the following "cuda", "cpu" or None
    if not (device is None or (type(device) == str and device.lower() in ["cuda",  "cpu"])):
        raise ValueError(f"device argument should take on of the following values ['cuda', 'cpu', None] but {device} is provided")
    
    #start processing the datasets. 
    ImageDatasetProcessor.process(
        input_folder,
        output_folder,
        clip_model, 
        pretrained, 
        batch_size, 
        num_threads, 
        device if device is None else device.lower(),
    )

     

if __name__ == "__main__": 
    
    fire.Fire(process_image_dataset_cli)