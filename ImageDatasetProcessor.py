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

class ImageDatasetProcessor: 
    """wrapper that contains the utility methods to process a dataset given its path. 
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
        """method to write file python dictionary into json file.  
        """
        
        with open(path, mode = 'w', encoding = 'utf-8') as json_file: 
            json.dump(to_write, json_file, indent = 4)
        
        
    @staticmethod
    def __image_metadata(image: Image.Image, image_id: int) -> Tuple[dict, int]: 
        """TODO 
        """
        
        image_path = os.path.abspath(image.filename)
        
        metadata = {
            'hash_id': ImageDatasetProcessor.__compute_blake2b(image), 
            'image_size_bytes': os.stat(image_path).st_size, 
            'path': image_path, 
            'name': os.path.split(image.filename)[1], 
            'type': image.format, 
        }
        
        return (metadata, image_id)

    @staticmethod
    def process_dataset(dataset_path: str, clip_model: str = 'ViT-B-32', pretrained: str = 'openai', num_threads: int = 4, batch_size: int = 32) -> None: 
        """TODO
        """
        #init the thread pool. 
        thread_pool = ThreadPoolExecutor(max_workers = num_threads)
        #detect the device to be used in calculating the embeddings. 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #load the CLIP model. 
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model,pretrained = pretrained)
        #load the image dataset. 
        images_loader = ImageDatasetLoader.load(dataset_path, recursive = True, batch_size = batch_size)

        json_result = {}
        # 

        with torch.no_grad(): 
            
        
            for images_chunk in images_loader: 
                                    
                preprocessed_chunk = torch.stack([preprocess(image) for image in images_chunk])

                tasks = [thread_pool.submit(ImageDatasetProcessor.__image_metadata, image, image_index,) for image_index, image in enumerate(images_chunk)]
                
                images_embeddings = model.encode_image(preprocessed_chunk.to(device))
                
                for task in as_completed(tasks): 
                    metadata, image_index = task.result() 
                    
                    metadata.update({
                        'clip_model': clip_model, 
                        'pretrained': pretrained,
                        'clip_embeddings': images_embeddings[image_index].tolist(), 
                    })
                    
                    json_result[metadata['hash_id']] = metadata
            
        
        json_path = "./outputs/clip-cache/{}.json".format(os.path.splitext(os.path.split(dataset_path)[1])[0])
        
        ImageDatasetProcessor.__write_to_json(json_result, json_path)
        
        thread_pool.shutdown() 

    @staticmethod
    def process(datasets_paths: list[str], clip_model: str = 'ViT-B-32', pretrained: str = 'openai', batch_size: int = 32, num_process: int = multiprocessing.cpu_count(), num_threads: int = 4) -> None: 
        """TODO 
        """
        
        os.makedirs('./outputs/clip-cache', exist_ok = True)
        process_pool = ProcessPoolExecutor(max_workers = num_process)
        
        
        for dataset_path in datasets_paths: 
            
            process_pool.submit(
                ImageDatasetProcessor.process_dataset,
                dataset_path,
                clip_model,
                pretrained,
                num_threads,
                batch_size
            )
            
        return 

def process_image_dataset_cli(datasets_paths: Union[list[str], str], directory_of_datasets: bool = False,  clip_model: str = 'ViT-B-32', pretrained: str = 'openai', batch_size: int = 32, num_process: int = multiprocessing.cpu_count(), num_threads: int = 4) -> None: 
    """TODO 
    """
    
    if directory_of_datasets: 
        if type(datasets_paths) == str:
            datasets_paths = os.listdir(datasets_paths)
        else: 
            raise TypeError("you should provide a single path as a string of directory full of datasets")
    
    ImageDatasetProcessor.process(
        datasets_paths, 
        clip_model, 
        pretrained, 
        batch_size, 
        num_process, 
        num_threads, 
    )

     

if __name__ == "__main__": 
    
    fire.Fire(process_image_dataset_cli)