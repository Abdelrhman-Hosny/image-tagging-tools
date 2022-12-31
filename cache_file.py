import os
import sqlite3
from zipfile import ZipFile
from PIL import Image
import hashlib

'''
Class Name: FileCache
Description: A class to create, add and fetch image data to .sqlite database file.
'''

class FileCache(object):
    '''Class Properties'''
    zips_info = []
    db_name = 'file_cache.sqlite'
    out_dir = './output'
    #db_path = ''

    def create_file_cache(self, out_dir=out_dir, db_name = db_name):
        '''
        Creating file-cache database
        '''
        db_path = f'{out_dir}/{db_name}'

        def __create_database(db_path):
            cmd = '''CREATE TABLE file_cache (
            file_name   TEXT    NOT NULL,
            path   TEXT            ,
            hash_id   TEXT          ,
            type            TEXT    ,
            is_archive   TEXT    ,
            n_content   INTEGER     ,
            container_archive   TEXT            
            );
            '''
            with sqlite3.connect(db_path) as conn:
                conn.execute('PRAGMA encoding="UTF-8";')
                conn.execute(cmd)
                conn.commit()
            # db = sqlite3.connect(db_path)
            # c = db.cursor()
            # c.execute('PRAGMA encoding="UTF-8";')
            # c.execute(cmd1)
            # db.commit()

        #make sure result output path exists 
        os.makedirs(out_dir, exist_ok = True) 
        # Check for existing file
        if(os.path.exists(db_path)):
            print (f'[ERROR]: Previous {db_name} already exist in {db_path}. File-cache creation stopped.')
            return None
        else:
            __create_database(db_path)
            print (f'[INFO]: database {db_path} created')

    def insert_file_to_cache(self, db_path, arg1, arg2, arg3, arg4, arg5):
        try:
            cmd = "insert into file_cache(file_name, path, hash_id, type, container_archive) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"', '"+arg5+"')"
            with sqlite3.connect(db_path) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            print ('[ERROR]: Insert file to cache failed, file cache database does not exist or might be in use!')

    def insert_zip_to_cache(self, db_path, arg1, arg2, arg3, arg4, arg5, arg6):
        try:
            cmd = "insert into file_cache(file_name, path, type, is_archive, n_content, container_archive) values ('"+arg1+"','"+arg2+"','"+arg3+"','"+arg4+"','"+arg5+"','"+arg6+"')"
            #cmd = "insert into file_cache(file_name, path, type, is_archive, n_content, container_archive) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"', '"+arg5+"', '"+arg6+"')"
            with sqlite3.connect(db_path) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            print ('[ERROR]: Insert file to cache failed, file cache database does not exist or might be in use!')

    def clear_cache(self, db_path, delete_cache = False):
        try:
            if delete_cache:
                # Delete file
                if(os.path.exists(db_path)):
                    os.remove(db_path)
                    print (f'[INFO] File-cache database {db_path} has been removed.')
            else:
                # Clear table only
                cmd = "DELETE FROM file_cache"
                with sqlite3.connect(db_path) as conn:
                    conn.execute(cmd)
                    conn.commit()
                print (f'[INFO] Table "file_cache" on {db_path} database has been cleared.')
        except Exception as e:
            print (f'[ERROR]: Clearing data from database failed, file cache database does not exist or might be in use!')

    def get_random_hash(self, db_path):
        try:
            cmd = "SELECT hash_id FROM file_cache ORDER BY RANDOM() LIMIT 1 ;"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {'hash_id':row[0]}
        except Exception as e:
            print (f'[ERROR]: {e} Getting random hash from cache failed, file cache database does not exist or might be in use!')

    def get_img_by_hash(self, db_path, hash_id=''):
        try:
            cmd = "SELECT * FROM file_cache WHERE hash_id = '"+hash_id+"'"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {
                            'file_name':row[0],
                            'file_path':row[1],
                            'hash_id':row[2],
                            'file_type':row[3],
                            'is_archive':row[4],
                            'n_content':row[5],
                            'container_archive':row[6],
                            }
        except Exception as e:
            print ('[ERROR]: Getting image data failed, file cache database does not exist or might be in use!')

    def get_random_image(self, db_path):
        try:
            cmd = "SELECT * FROM file_cache ORDER BY RANDOM() LIMIT 1 ;"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {
                            'file_name':row[0],
                            'file_path':row[1],
                            'hash_id':row[2],
                            'file_type':row[3],
                            'is_archive':row[4],
                            'n_content':row[5],
                            'container_archive':row[6],
                            }
        
        except Exception as e:
            print ('[ERROR]: Getting image data failed, file cache database does not exist or might be in use!')


    def data_gen(self, data_file):
        '''Image generator for zip file'''
        
        global zips_info

        if data_file.endswith('.zip'):
            # Selected data_dir is a zip archive

            print (f'[INFO] Processing ZIP archive: {data_file}')
            
            with ZipFile(data_file) as archive:

                '''Getting archive details'''
                # Check the number of content (image file)
                entries = archive.infolist()
                n_content =  len([content for content in entries if content.is_dir() ==False])
                # Appending to the list to be writen to database table
                self.zips_info.append([data_file, n_content])

                for entry in entries:
                    # Do for every content in the zip file
                    if not entry.is_dir():
                        
                        with archive.open(entry) as file:

                            if entry.filename.lower().endswith(('.zip')):
                                # Another zip file found in the content.
                                print (f'[INFO] Processing ZIP archive: {data_file}/{entry.filename}')
                                # Process the content of the zip file
                                with ZipFile(file) as sub_archive:

                                    '''Getting archive details'''
                                    # Check the number of content
                                    sub_entries = sub_archive.infolist()
                                    n_content =  len([content for content in sub_entries if content.is_dir() ==False])
                                    # Appending to the list to be writen to database table
                                    self.zips_info.append([f'{data_file}/{entry.filename}', n_content])

                                    for sub_entry in sub_entries:
                                        with sub_archive.open(sub_entry) as sub_file:
                                            try:
                                                img = Image.open(sub_file)
                                                img_file_name = f'{data_file}/{entry.filename}/{sub_entry.filename}'
                                                print (f' Fetching: {img_file_name}')
                                                yield (img, img_file_name)
                                            except:
                                                print (f'[WWARNING] Failed to fetch {os.path.join(data_file, entry.filename, sub_entry.filename)}')
                                                continue
                            else:
                                # Should be image file. Read it.
                                try:
                                    img = Image.open(file)
                                    img_file_name = f'{data_file}/{entry.filename}'
                                    print (f' Fetching: {img_file_name}')
                                    yield (img, img_file_name)
                                except:
                                    print (f'[WARNING] Failed to fetch {data_file}/{entry.filename}')
                                    continue
        else:
            # Should be image file. Read it.
            try:
                img = Image.open(data_file)
                print (f' Fetching: {data_file}')
                yield (img, data_file)
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
                print(f"[ERROR]  cannot compute hash for {img_file_name} , {e}")
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

    def add_folder_to_file_cache(self, data_dir, out_dir=out_dir, db_name = db_name):
        '''Load images to file-cache'''
        
        db_path = f'{out_dir}/{db_name}'

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
                
                # Check if it is container archive
                container_archive = ''
                if file.lower().endswith(('.zip')):
                    container_archive = file

                for img, file_path in self.data_gen(file):
                    file_name = os.path.basename(file_path)
                    type = os.path.splitext(file_name)[-1]
                    #tag_folder = os.path.split(os.path.splitext(os.path.split(file_path)[0])[0])[-1]
                    # Compute hash
                    hash = self.compute_hash(img, file_path)  
                    # Insert image to cache
                    self.insert_file_to_cache(db_path, file_name, file_path, hash, type, container_archive)  

            # Insert zip files to cache
            for _zip in self.zips_info:
                parent = os.path.split(_zip[0])[0]
                if parent.lower().endswith(('.zip')):
                    container_archive = parent
                else:
                    pass
                    #parent_folder = parent
                file_name = os.path.basename(_zip[0])
                arch_path = _zip[0]
                n_content = _zip[1]
                self.insert_zip_to_cache(db_path, file_name, arch_path, os.path.splitext(file_name)[-1], str(True), str(n_content), container_archive)

        else:
            print (f'[ERROR] Database {db_path} does not exist !')

        print (f'[INFO]: Folder {data_dir} added to {db_path}')