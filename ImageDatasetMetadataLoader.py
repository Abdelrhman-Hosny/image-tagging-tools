import time
from typing import Union
import simdjson
import os 

class ImageDatasetMetadataLoader:
    """class to load and manipulate datasets metadata saved into json files.
    """
    def __init__(self) -> None:
        
        self.parser = simdjson.Parser()
        self.datasets_metadata_map = {} 
    
    def load_json(self, path: str) -> Union[dict, list]:
        """loads a json file given its path. 
            
        :param path: path to the json file. 
        :type path: str
        :returns: returns the file loaded into Python Objects. 
        :rtype: Union[dict, list]
        """
        
        return self.parser.load(path, recursive = True)
    
    def load_directory(self, path: str = './outputs/clip-cache') -> dict: 
        """loads all json files of image datasets metadata for a given directory. 
            
        :param path: path to the directory. 
        :type path: str
        :returns: returns the all files concatenated and loaded into Python Objects. 
        :rtype: dict
        """
        
        #get all files in the provided directory. 
        files = [os.path.join(root, file) for root, folders, files in os.walk(path) for file in files]
        #filter JSON files only. 
        json_files = [file for file in files if os.path.splitext(file)[1].lower() == '.json']
                
        #loop over the JSON files. 
        for file_path in json_files:
            dataset_metadata = self.load_json(file_path) #load the json file into dict. 
            
            #categorize the data by model and then the image hash. 
            for image_hash, image_metadata in dataset_metadata.items(): 
                model = image_metadata['model']
                
                if model not in self.datasets_metadata_map: 
                    self.datasets_metadata_map[model] = {}
                
                self.datasets_metadata_map[model][image_hash] = image_metadata
            

        return self.datasets_metadata_map

if __name__ == "__main__": 
    
    start_time = time.time()
    json_loader = ImageDatasetMetadataLoader()
    
    json_loader.load_directory()
    
    print(time.time() - start_time)