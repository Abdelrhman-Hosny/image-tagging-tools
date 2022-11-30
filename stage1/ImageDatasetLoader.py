import os
from typing import Iterator 
from PIL import Image
import patoolib
import shutil
import glob

class ImageDatasetLoader: 
    """utility methods to load datasets of images from folder or an archive file given the folder or the archive file path. 
    """
    def __init__(self) -> None:
        pass
    @staticmethod
    def __list_dir(dir_path: str, recursive: bool = True): 
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
    def __latest_in_folder(path: str) -> str:
        """find the latest file/folder in a specific folder.

        :param path: path of the folder to search in.
        :type path: str
        :returns: path of the latest file/folder.
        :rtype: str
        """

        list_of_files = glob.glob(f'{path}/*') # * means all 
        latest_file = max(list_of_files, key=os.path.getctime)

        return latest_file
        
        
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
        
        output_directory = os.path.join('./outputs/tmp' , output_path)
        
        #make sure the output dir is found or else create it. 
        os.makedirs(output_directory, exist_ok = True)
        patoolib.extract_archive(path, outdir = output_directory)
        
        file_name = os.path.split(ImageDatasetLoader.__latest_in_folder(output_directory))[1]

        return os.path.join(output_directory, file_name) # modified return value
    
    @staticmethod
    def convert_gif_to_image(gif_path:str):
        """Delets the GIF and change it with first frame .png image
        :param gif_path: path to the GIF file.
        :type gif_path: str
        :rtype: None
        """
        im = Image.open(gif_path)
        dir_path = os.path.dirname(os.path.abspath(gif_path))
        im.seek(0)
        im_file = os.path.basename(gif_path).split('.gif')[0]
        save_path = os.path.join(dir_path ,f'{im_file}.png' )
        im.save(save_path) # save the first frame as .png image 
        im.close()
        os.remove(gif_path)
        #os.system(f'rm -r {gif_path}') # Delete the .gif file 

    @staticmethod
    def clean_file(file_path: str):
        """This function takes a file path and see if it is supported or not. 
            :param file_path: path of the file to work with 
            :type file_path: str
        """
        if file_path.lower().endswith('.gif'): # If it's GIF then convert to image and exit 
            try : 
                ImageDatasetLoader.convert_gif_to_image(file_path)
            except Exception as e:
                print(f"[Warning] problem with {file_path}, {e}")
            if os.path.exists(file_path):
                print(f"[Warning] removing {file_path}")
                os.remove(file_path)
                #os.system(f'rm {file_path}')
            return 

    @staticmethod
    def clean_directory(dir_path: str, only_sub_dir: bool = False):
        """ clean a directory files and folders

        :param dir_path: path to the directory which will be cleaned.
        :type dir_path: str
        :param only_sub_dir: an option to make it only sub directories 
                            for ex: in cleaning pixel-art-tagged folder.
        :type only_sub_dir: bool
        :rtype: None
        """
        
        for dir in os.listdir(dir_path):
            sub_dir = os.path.join(dir_path, dir)
            
            if os.path.isfile(sub_dir): # if it's a file then clean the file  
                if only_sub_dir: # no subfiles allowed for example in the pixel-art-tagged 
                    os.remove(sub_dir)
                    print(f"[Removing] {sub_dir}")
                    continue 
                
                ImageDatasetLoader.clean_file(sub_dir)
                continue

            if len(os.listdir(sub_dir)) == 0: # Empty folder, delte it 
                shutil.rmtree(sub_dir)
                #os.system(f'rm -r {sub_dir}') 
                print(f'[Removing] {sub_dir}')
                continue

            if os.path.isdir(sub_dir) and only_sub_dir: # move to the sub-directory and clean it.
                ImageDatasetLoader.clean_directory(sub_dir)
            else:
                shutil.rmtree(sub_dir)  

    @staticmethod
    def load(dataset_path: str, recursive: bool = True, batch_size: int = 32): 
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
            print("is archive dataset")
            print(f"dataset folder path  = {image_dataset_folder_path}")
        
        #clean the dataset
        ImageDatasetLoader.clean_directory(image_dataset_folder_path, only_sub_dir=True)
        #get all tags in the dataset. 
        tags = [tag.lower() for tag in os.listdir(image_dataset_folder_path)]
        print(tags)
        
        #make sure other-training and other-validation tags are available. 
        
#        error = False 
#        if "other-training" not in tags or len(os.listdir(os.path.join(image_dataset_folder_path, "other-training"))) == 0:
#            error = "`other-training` folder should be contained in the dataset and not empty"
#        
#        if "other-validation" not in tags or len(os.listdir(os.path.join(image_dataset_folder_path, "other-validation"))) == 0: 
#            error = "`other-validation` folder should be contained in the dataset and not empty"
#
#        if  error is not False: 
#            if archive_dataset: 
#                shutil.rmtree(image_dataset_folder_path)
#            raise AssertionError(error)
            
        
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
                    print(f"[WARNING]: image {file_path} was be skipped due to the error {error}")
                    continue
            
            #it's the last element, as it's generator to avoid error when deleting the folder and the file is accessed by another process.
            if archive_dataset and last_chunk:
                yield images.copy()
            else: 
                yield images

        if archive_dataset:
            shutil.rmtree(image_dataset_folder_path)
