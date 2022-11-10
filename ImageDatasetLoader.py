import os
from typing import Iterator 
from PIL import Image
import patoolib
import shutil

class ImageDatasetLoader: 
    """utility methods to load datasets of images from folder or an archive file given the folder or the archive file path. 
    """
    def __init__(self) -> None:
        pass
    @staticmethod
    def __list_dir(dir_path: str, recursive: bool = True) -> list[str]: 
        """method to list all file paths for a given directory. 
        :param dir_path: The directory to get the it's files paths
        :type dir_path: str
        :param recursive: If it's set to True the function will return paths of all files in the given directory 
                and all its subdirectories
        :type recursive: bool
        :returns: list of files
        :rtype: list[str]
        """
        
        if recursive:
            return [os.path.join(root, file) for root, folders, files in os.walk(dir_path) for file in files]
        else: 
            return [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    
    @staticmethod
    def __is_archive(path: str) -> bool: 
        """method to check if a given path is an archive.
        :param path: The file path to check. 
        :type path: str
        :returns: `True` if the given path is a path of an archived file. 
        :rtype: bool
        """
        
        try: 
            patoolib.get_archive_format(path)
            return True 
        except Exception: 
            return False 
        
    @staticmethod
    def __extract_archive(path: str) -> str: 
        """method to decompress an archive given its path. 
        
        :param path: The archive path to decompress. 
        :type path: str
        :returns: decompresses the archive and returns the path of the extracted folder. 
        :rtype: str
        """
        
        root_path, file_name = os.path.split(path)

        file_name, _ = os.path.splitext(file_name)
        
        output_path = f"{file_name}-decompressed-tmp"
        
        #make sure the output dir is found or else create it. 
        os.makedirs(output_path, exist_ok = True)

        patoolib.extract_archive(path, outdir = os.path.join(root_path, output_path))
        
        return output_path

    @staticmethod
    def load(dataset_path: str, recursive: bool = True) -> Iterator[Image.Image]: 
        """loader for the given dataset path, it returns a generator 
        
        :param dataset_path: path of the dataset either it's an archive or a directory of images.
        :type dataset_path: str
        :param recursive: If it's set to True the function will return paths of all files in the given directory 
                and all its subdirectories
        :type recursive: bool
        :returns: an iterator of images in the folder. 
        :rtype: Iterator[PIL.Image.Image]
        """
        
        archive_dataset = False 
        image_dataset_folder_path = dataset_path 
        # if the given path is a path of an archive. 
        if ImageDatasetLoader.__is_archive(dataset_path):
            archive_dataset = True 
            image_dataset_folder_path = ImageDatasetLoader.__extract_archive(dataset_path)

        dataset_files_paths = ImageDatasetLoader.__list_dir(image_dataset_folder_path, recursive)
        #loop over the files list of the folder. 
        for index, file_path in enumerate(dataset_files_paths):

            try: #try to open file path as image
                image = None 
                #it's the last element, as it's generator to avoid error when deleting the folder and the file is accessed by another process.
                if archive_dataset and index == len(dataset_files_paths) - 1:
                    image = Image.open(file_path).copy()
                else: 
                    image = Image.open(file_path)
                
                yield image #ok file is image
            
            except Exception: #file is not a valid image.  
                continue
        
        if archive_dataset:
            shutil.rmtree(image_dataset_folder_path)
