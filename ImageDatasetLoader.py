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
        
        output_directory = os.path.join('./outputs/tmp', output_path)
        #make sure the output dir is found or else create it. 
        os.makedirs(output_directory, exist_ok = True)

        patoolib.extract_archive(path, outdir = output_directory)
        
        return output_directory

    @staticmethod
    def load(dataset_path: str, recursive: bool = True, batch_size: int = 32) -> Iterator[list[Image.Image]]: 
        """loader for the given dataset path, it returns a generator 
        
        :param dataset_path: path of the dataset either it's an archive or a directory of images.
        :type dataset_path: str
        :param recursive: If it's set to True the function will return paths of all files in the given directory 
                and all its subdirectories
        :type recursive: bool
        :returns: an iterator of images in the folder. 
        :rtype: Iterator[list[PIL.Image.Image]]
        """
        
        archive_dataset = False 
        image_dataset_folder_path = dataset_path 
        # if the given path is a path of an archive. 
        if ImageDatasetLoader.__is_archive(dataset_path):
            archive_dataset = True 
            image_dataset_folder_path = ImageDatasetLoader.__extract_archive(dataset_path)
            
            #get all tags in the dataset. 
            tags = [tag.lower() for tag in os.listdir(image_dataset_folder_path)]
            
            #make sure other-training and other-validation tags are available. 
            error = False 
            if "other-training" not in tags or len(os.listdir(os.path.join(image_dataset_folder_path, "other-training"))) == 0:
                error = "`other-training` folder should be contained in the dataset and not empty"
            
            if "other-validation" not in tags or len(os.listdir(os.path.join(image_dataset_folder_path, "other-validation"))) == 0: 
                error = "`other-validation` folder should be contained in the dataset and not empty"

            if error is not False: 
                shutil.rmtree(image_dataset_folder_path)
                raise AssertionError(error)
            
        
        dataset_files_paths = ImageDatasetLoader.__list_dir(image_dataset_folder_path, recursive)
        #loop over the files list of the folder. 
        
        for chunk_pos in range(0, len(dataset_files_paths), batch_size):

            files_chunk = dataset_files_paths[chunk_pos: min(chunk_pos + batch_size, len(dataset_files_paths))]
            
            last_chunk = (chunk_pos + batch_size >= len(dataset_files_paths))
            
            images = [] 
            
            for file_path in files_chunk: 
                try: #try to open file path as image
                    images.append(Image.open(file_path))
                except Exception as error: #file is not a valid image.
                    print(f"[WARNING]: image {file_path} was will be skipped due to the error {error}")
                    continue
            
            #it's the last element, as it's generator to avoid error when deleting the folder and the file is accessed by another process.
            if archive_dataset and last_chunk:
                yield images.copy()
            else: 
                yield images

        if archive_dataset:
            shutil.rmtree(image_dataset_folder_path)
