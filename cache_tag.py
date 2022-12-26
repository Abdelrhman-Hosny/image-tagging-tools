import os
import sqlite3
import os
import sqlite3
from zipfile import ZipFile
from PIL import Image
import hashlib

'''
Class Name: TagCache
Description: A class to create .sqlite database file and add, fetch and clear image tag data to it.
'''

# Takes directory that contains dataset images
# Lists every image file in the dataset directory
# For every file, compute the hash, find its tag based on the tag folder that contains it.
# Write the respective tag to the tag_cache.sqlite in the specified output directory.

class TagCache(object):

    '''Class Properties'''
    zips_info = []
    db_name = 'tag_cache.sqlite'
    out_dir = './output'

    def create_tag_cache(self, out_dir=out_dir, db_name = db_name):
        '''
        Creating tag-cache database
        '''
        db_path = f'{out_dir}/{db_name}'

        def __create_database(db_path):
            cmd = '''CREATE TABLE tag_cache (
            hash_id     TEXT,
            tag     TEXT            
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

    def insert_tag_to_cache(self, db_path, arg1, arg2):
        try:
            cmd = "insert into tag_cache(hash_id, tag) values ('"+arg1+"', '"+arg2+"')"
            with sqlite3.connect(db_path) as conn:
                conn.execute(cmd)
                conn.commit()
        except Exception as e:
            print (f'[ERROR] {e}: Insert file to cache failed, file cache database does not exist or might be in use!')

    def clear_cache(self, db_path, delete_cache = False):
        try:
            if delete_cache:
                # Delete file
                if (os.path.exists(db_path)):
                    os.remove(db_path)
                    print (f'[INFO] Tag-cache database {db_path} has been removed.')
            else:
                # Clear table only
                cmd = "DELETE FROM tag_cache"
                with sqlite3.connect(db_path) as conn:
                    conn.execute(cmd)
                    conn.commit()
                print (f'[INFO] Table "tag_cache" on {db_path} database has been cleared.')
        except Exception as e:
            print (f'[ERROR] {e}: Clearing data from database failed, tag cache database does not exist or might be in use!')

    def get_random_hash(self, db_path):
        try:
            cmd = "SELECT hash_id FROM tag_cache ORDER BY RANDOM() LIMIT 1 ;"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {'hash_id':row[0]}
        except Exception as e:
            print (f'[ERROR] {e}: Getting random hash from cache failed, tag cache database does not exist or might be in use!')

    def get_tag_by_hash(self, db_path, hash_id=''):
        try:
            cmd = "SELECT * FROM tag_cache WHERE hash_id = '"+hash_id+"'"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {
                            'hash_id':row[0],
                            'tag':row[1],
                            }
        except Exception as e:
            print (f'[ERROR] {e}: Getting tag failed, tag cache database does not exist or might be in use!')

    def get_random_tag(self, db_path):
        try:
            cmd = "SELECT * FROM tag_cache ORDER BY RANDOM() LIMIT 1 ;"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    return {
                            'hash_id':row[0],
                            'tag':row[1],
                            }
        
        except Exception as e:
            print (f'[ERROR] {e}: Getting tag failed, tag cache database does not exist or might be in use!')


    def data_gen(self, data_file):
        '''Image generator'''
        
        if data_file.endswith('.zip'):
            # Selected data_dir is a zip archive
            
            with ZipFile(data_file) as archive:

                '''Getting archive details'''
                # Check the number of content (image file)
                entries = archive.infolist()

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
                                    sub_entries = sub_archive.infolist()

                                    for sub_entry in sub_entries:
                                        with sub_archive.open(sub_entry) as sub_file:
                                            try:
                                                img = Image.open(sub_file)
                                                img_file_name = f'{data_file}/{entry.filename}/{sub_entry.filename}'
                                                print (f' Processing: {img_file_name}')
                                                yield (img, img_file_name)
                                            except:
                                                print (f'[WARNING] Failed to process {os.path.join(data_file, entry.filename, sub_entry.filename)}')
                                                continue
                            else:
                                # Should be image file. Read it.
                                try:
                                    img = Image.open(file)
                                    img_file_name = f'{data_file}/{entry.filename}'
                                    print (f' Processing: {img_file_name}')
                                    yield (img, img_file_name)
                                except:
                                    print (f'[WARNING] Failed to process {data_file}/{entry.filename}')
                                    continue
        else:
            # Should be image file. Read it.
            try:
                img = Image.open(data_file)
                print (f' Processing: {data_file}')
                yield (img, data_file)
            except:
                print (f'[WARNING] Failed to process {data_file}')
        

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

    def add_folder_to_tag_cache(self, data_dir, out_dir=out_dir, db_name = db_name):
        '''Load images to tag-cache'''
        
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
                        files_list.append(f'{root}/{file}')
            else:
                '''Single file (could be a zip archive or image)'''
                files_list = [data_dir]

            for file in files_list:
                '''Processing images'''
            
                for img, file_path in self.data_gen(file):
                    # Consider only image files within tag folders. Its root != data_dir 
                    #print (f'FILE: {file}, {file_path}, {data_dir}')
                    if (os.path.split(file_path)[0].strip() != data_dir.strip()):
                        tag_folder = os.path.split(os.path.splitext(os.path.split(file_path)[0])[0])[-1]
                        # Compute hash
                        hash = self.compute_hash(img, file_path)  
                        # Insert image to cache
                        self.insert_tag_to_cache(db_path, hash, tag_folder)  
                    else:
                        print(f'[WARNING] : File {file_path} is outside of tag folder. Ignoring it...')

        else:
            print (f'[ERROR] Database {db_path} does not exist !')

        print (f'[INFO]: Folder {data_dir} added to {db_path}')