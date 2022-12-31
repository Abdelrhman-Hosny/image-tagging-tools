import os
import sqlite3
from zipfile import ZipFile
from PIL import Image
import hashlib
import torch
import open_clip
import pickle

'''
Class Name: ClipCache
Description: A class to create, add and fetch clip vector data to .sqlite database file.
'''

class ClipCache(object):
    '''Class Properties'''

    db_name = 'clip_cache.sqlite'
    out_dir = './output'
    clip_model_type = 'ViT-B-32'
    clip_model_pretrained = 'openai'


    def create_clip_cache(self, out_dir=out_dir, db_name = db_name):
        '''
        Creating clip-cache database
        '''
        db_path = f'{out_dir}/{db_name}'

        def __create_database(db_path):
            cmd = '''CREATE TABLE clip_cache (
            hash_id   TEXT          ,
            clip_vector   BLOB      ,
            model   TEXT      
            );
            '''
            with sqlite3.connect(db_path) as conn:
                conn.execute('PRAGMA encoding="UTF-8";')
                conn.execute(cmd)
                conn.commit()

        #make sure result output path exists 
        os.makedirs(out_dir, exist_ok = True) 
        # Check for existing file
        if(os.path.exists(db_path)):
            print (f'[ERROR]: Previous {db_name} already exist in {db_path}. File-cache creation stopped.')
            return None
        else:
            __create_database(db_path)
            print (f'[INFO]: database {db_path} created')

    def insert_clip_to_cache(self, db_path, arg1, arg2, arg3):
        try:
            cmd = """insert into clip_cache(hash_id, clip_vector, model) values (?, ?, ?)"""
            with sqlite3.connect(db_path) as conn:
                conn.execute(cmd, (arg1, arg2, arg3))
                conn.commit()
        except Exception as e:
            print (f'[ERROR] {e}: Insert clip to cache failed, clip cache database does not exist or might be in use!')

    def clear_cache(self, db_path, delete_cache = False):
        try:
            if delete_cache:
                # Delete file
                if(os.path.exists(db_path)):
                    os.remove(db_path)
                    print (f'[INFO] Clip-cache database {db_path} has been removed.')
            else:
                # Clear table only
                cmd = "DELETE FROM clip_cache"
                with sqlite3.connect(db_path) as conn:
                    conn.execute(cmd)
                    conn.commit()
                print (f'[INFO] Table "clip_cache" on {db_path} database has been cleared.')
        except Exception as e:
            print (f'[ERROR] {e}: Clearing data from database failed, clip cache database does not exist or might be in use!')

    def get_random_hash(self, db_path):
        try:
            cmd = "SELECT hash_id FROM clip_cache ORDER BY RANDOM() LIMIT 1 ;"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {'hash_id':row[0]}
        except Exception as e:
           print (f'[ERROR] {e}: Getting random hash from cache failed, clip cache database does not exist or might be in use!')

    def get_clip_by_hash(self, db_path, hash_id=''):
        try:
            cmd = "SELECT * FROM clip_cache WHERE hash_id = '"+hash_id+"'"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {
                            'hash_id':row[0],
                            'clip_vector': pickle.loads(row[1]),
                            'model' : row[2]
                            }
        except Exception as e:
            print (f'[ERROR] {e}: Getting clip vector failed, clip cache database does not exist or might be in use!')

    def get_random_clip(self, db_path):
        try:
            cmd = "SELECT * FROM clip_cache ORDER BY RANDOM() LIMIT 1 ;"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {
                            'hash_id':row[0],
                            'clip_vector': pickle.loads(row[1]),
                            'model' : row[2]
                            }
        
        except Exception as e:
            print (f'[ERROR] {e}: Getting clip vector failed, clip cache database does not exist or might be in use!')


    def data_gen(self, data_file):
        '''Image generator for data_file'''
        
        if data_file.endswith('.zip'):
            # Selected data_dir is a zip archive
            
            with ZipFile(data_file) as archive:

                # Getting archive details
                entries = archive.infolist()

                for entry in entries:
                    # Do for every content in the zip file
                    if not entry.is_dir():
                        
                        with archive.open(entry) as file:

                            if entry.filename.lower().endswith(('.zip')):
                                # Another zip file found in the content. Process the content of the zip file
                                with ZipFile(file) as sub_archive:

                                    '''Getting archive details'''
                                    # Check the number of content
                                    sub_entries = sub_archive.infolist()

                                    for sub_entry in sub_entries:
                                        with sub_archive.open(sub_entry) as sub_file:
                                            try:
                                                img = Image.open(sub_file)
                                                yield (img)
                                            except:
                                                print (f'[WWARNING] Failed to fetch {os.path.join(data_file, sub_entry.filename)}')
                                                continue
                            else:
                                # Should be image file. Read it.
                                try:
                                    img = Image.open(file)
                                    yield (img)
                                except:
                                    print (f'[WARNING] Failed to fetch {entry.filename}')
                                    continue
        else:
            # Should be image file. Read it.
            try:
                img = Image.open(data_file)
                print (f' Fetching: {data_file}')
                yield (img)
            except:
                print (f'[WARNING] Failed to fetch {data_file}')
        

    def compute_hash(self, img, img_file_name):
        '''Compute image file to hash'''
        if img_file_name.lower().endswith('.gif'): # If it's GIF then convert to image and exit 
            try : 
                # Convert gif to image
                img.seek(0)
                # Compute hash
                return hashlib.blake2b(img.tobytes()).hexdigest()
            except Exception as e:
                print(f"[ERROR] {e}:  cannot compute hash for {img_file_name}")
                return None 
        return hashlib.blake2b(img.tobytes()).hexdigest()

    def empty_dirs_check(self, dir_path):
        """ Checking for empty directory and print out warning if any"""
        for dir in os.listdir(dir_path):
            sub_dir = os.path.join(dir_path, dir)
            # Check for directory only
            if os.path.isdir(sub_dir):
                if len(os.listdir(sub_dir)) == 0:
                    # Empty folder
                    print(f'[WARNING] Empty folder found. Ignoring it: {sub_dir}')
                    continue

    def get_clip(self, clip_model_type : str = 'ViT-B-32', pretrained : str = 'openai'):
        # get clip model from open_clip
        clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_type,pretrained=pretrained)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return clip_model, preprocess, device

    def get_clip_vector(self, img, image_file_name : str, model, preprocess, device:str):
        with torch.no_grad():
            
            if image_file_name.lower().endswith('.gif'): 
                try:
                    img.seek(0)
                except:
                    print [f'[WARNING] Failed to convert {image_file_name} image.']
                    return
            else:
                # Image files other than gif
                img_obj = img

            image = preprocess(img_obj).unsqueeze(0).to(device)
            return model.encode_image(image).detach().numpy()


    def add_folder_to_clip_cache(self, data_dir, out_dir=out_dir, db_name = db_name):
        '''Load images to clip-cache'''
        
        # Setting the database path
        db_path = f'{out_dir}/{db_name}'
        # Getting clip model
        clip_model , preprocess , device = self.get_clip(clip_model_type= self.clip_model_type, pretrained= self.clip_model_pretrained)

        if (os.path.exists(db_path)):

            # Placeholder for data file names
            files_list = []
            
            if not os.path.isfile(data_dir):
                '''For normal directory'''
                # Check for empty dirs
                self.empty_dirs_check(data_dir)
                # Walking thru files
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        files_list.append(os.path.join(root, file))
            else:
                '''Single file (could be a zip archive or image)'''
                files_list = [data_dir]

            
            for file in files_list:
                '''Fetching images'''
            
                for img in self.data_gen(file):
                    # Compute hash
                    hash = self.compute_hash(img, file)  
                    print (f'[INFO] Calculating CLIP vector for {file}...')
                    # Compute clip vector
                    clip_vector = self.get_clip_vector(img, file, clip_model,preprocess,device)
                    clip_vector = pickle.dumps(clip_vector)
                    # Insert image to cache
                    self.insert_clip_to_cache(db_path, hash, clip_vector, f'{self.clip_model_type}:{self.clip_model_pretrained}')  

        else:
            print (f'[ERROR] Database {db_path} does not exist !')

        print (f'[INFO]: Folder {data_dir} added to {db_path}')