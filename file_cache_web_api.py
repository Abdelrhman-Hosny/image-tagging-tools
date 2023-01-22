import os
import argparse
import sqlite3
import io
from zipfile import ZipFile
from PIL import Image
import hashlib
import torch
import open_clip
from flask import Flask, request, render_template
import json
import pickle
import base64
import time


'''File Cache Web API'''

# Server
app = Flask(__name__)

def get_img_data(file_path):
    '''Fetch PIL image object from image file under directory or ZIP archive'''

    # Placeholder for directories and files in file path
    path_list = []

    # Walk thru file_paths and append the path_list
    head = file_path
    while True:
        head, tail = os.path.split(head)
        path_list.append(tail)
        if head == '' or head == '.':
            # Arrive at root. Break the loop
            break
    # Reverse so that index 0 is the top of file path.
    path_list=path_list[::-1]

    # Walk from top of file path and return the PIL image object once it find image file  
    # Placeholder for walked path
    walk_path= ''

    for i in range(len(path_list)):

        walk_path = os.path.join(walk_path, path_list[i])
        
        # Check if it is a file
        if os.path.isfile(walk_path):
            
            # Check if it is a ZIP archive or image file
            if walk_path.lower().endswith(('.zip')):
                
                # ZIP archive, open the content
                with ZipFile(walk_path) as archive:
                    
                    # Walk starting form ZIP archive to the rest of the path in the path_list
                    zip_walk_path = ''
                    for j in range(i+1, len(path_list)):

                        try:
                            zip_walk_path = os.path.join(zip_walk_path, path_list[j])
                            # Convert to unix like path. Unless getinfo will raise an error
                            zip_walk_path = zip_walk_path.replace('\\','/')
                            # getinfo will raise error except on valid file (image or child ZIP archive)
                            entry = archive.getinfo(zip_walk_path)

                            # Check if it is a file
                            if not entry.is_dir():

                                with archive.open(entry) as file:

                                    if entry.filename.lower().endswith(('.zip')):
                                        # Child ZIP archive. Open it

                                        with ZipFile(file) as sub_archive:
                                            
                                            # Walk from child ZIP archive to the rest of the path
                                            child_zip_walk_path = ''
                                            for k in range(j+1, len(path_list)):
                                                try:
                                                    child_zip_walk_path = os.path.join(child_zip_walk_path, path_list[k])
                                                    child_zip_walk_path = child_zip_walk_path.replace('\\','/')
                                                    sub_entry = sub_archive.getinfo(child_zip_walk_path)
                                                    
                                                    if not entry.is_dir():
                                                        with sub_archive.open(sub_entry) as sub_file:
                                                            img = Image.open(sub_file)
                                                            img_size = img.size
                                                            img_byte = io.BytesIO()
                                                            img.save(img_byte, 'PNG')
                                                            return True, img_byte, img_size
                                                except:
                                                    return False, None
                                    
                                    else:
                                        # Should be image file. Read it.
                                        try:
                                            img = Image.open(file)
                                            img_size = img.size
                                            img_byte = io.BytesIO()
                                            img.save(img_byte, 'PNG')
                                            return True, img_byte, img_size
                                        except:
                                            return False, None
                        
                        except Exception as e:
                            # getinfo() have ont arrive in the valid file yet. Continue walking on the path
                            # print (e)
                            continue

            else:
                # Should be image file. Read it.
                try:
                    img = Image.open(walk_path)
                    img_size = img.size
                    img_byte = io.BytesIO()
                    img.save(img_byte, 'PNG')
                    return True, img_byte, img_size
                except:
                    return False, None
            
        else:
            # It is a directory, do nothing and continue.
            continue

    # Unable to open any image file
    return False, None


@app.route('/get_random_img')
def get_random_img():

    try:
        
        # Start the request timing
        t1 = time.time()

        # Argument extraction from URL query string
        _db_path = request.args.get('db_path')        

        if _db_path:
            db_path = _db_path
        else:
            raise Exception ('Database path not specified')

        cmd = "SELECT * FROM file_cache ORDER BY RANDOM() LIMIT 1 ;"

        while True:

            # Loop until getting valid image file (not ZIP archive)
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    file_name=row[0]
                    file_path=row[1]
                    hash_id=row[2]
                    file_type=row[3]
                    is_archive=row[4]
                    container_archive=row[6]

                if not is_archive:
                    # If it is ZIP archive, repeat the random selection until getting image file.
                    break
        
        # Get the image content (byte data) from specified file_path
        ret, img_byte, img_size = get_img_data(file_path)

        if ret:
            # Image data is successfully retrieved
            img_data = base64.b64encode(img_byte.getvalue()).decode('utf-8')
            # Calculate the request time
            t2 = time.time()
            request_time = f'{round((t2-t1)*1000, 3)} ms'
            # Rendering
            return render_template('web_template.html',
                                    img_data = img_data, 
                                    file_name = file_name,
                                    file_path = file_path,
                                    img_size = str(img_size),
                                    container_archive = container_archive,
                                    hash_id = hash_id,
                                    request_time = request_time
                                    )
        else:
            raise Exception(f'Unable to read {file_path}')
    
    except Exception as e:
        return (f'[ERROR] {e}: Getting image data failed')


'''
The following are disabled functions related to creation, adding data, clearing data, fetching random data from 
file cache, CLIP cache and tag cache.
'''
# '''File Cache'''

# zips_info = []

# app = Flask(__name__)

# @app.route('/create_file_cache')
# def create_file_cache():
#     '''
#     Creating file-cache database
#     '''
#     # Default
#     out_dir= './output'
#     db_name = 'file_cache.sqlite'

#     # Argument extraction from URL query string
#     _out_dir = request.args.get('out_dir')
#     _db_name = request.args.get('db_name')

#     if _out_dir:
#         out_dir = str(_out_dir)
#     if _db_name:
#         db_name = str(_db_name)

#     db_path = f'{out_dir}/{db_name}'

#     def __create_database(db_path):
#         cmd = '''CREATE TABLE file_cache (
#         file_name   TEXT    NOT NULL,
#         path   TEXT            ,
#         hash_id   TEXT          ,
#         type            TEXT    ,
#         is_archive   TEXT    ,
#         n_content   INTEGER     ,
#         container_archive   TEXT            
#         );
#         '''
#         with sqlite3.connect(db_path) as conn:
#             conn.execute('PRAGMA encoding="UTF-8";')
#             conn.execute(cmd)
#             conn.commit()

#     try:
#         # Make sure result output path exists 
#         os.makedirs(out_dir, exist_ok = True) 
#         # Check for existing file
#         if(os.path.exists(db_path)):
#             return (f'[ERROR]: Previous {db_name} already exist in {db_path}. File-cache creation stopped.')
#         else:
#             __create_database(db_path)
#             return (f'[OK]: database {db_path} created')
#     except Exception as e:
#         return (f'[ERROR] {e}: Creating cache database failed!')


# def insert_file_to_cache(db_path, arg1, arg2, arg3, arg4, arg5):
#     try:
#         cmd = "insert into file_cache(file_name, path, hash_id, type, container_archive) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"', '"+arg5+"')"
#         with sqlite3.connect(db_path) as conn:
#             conn.execute(cmd)
#             conn.commit()
#     except Exception as e:
#         return (f'[ERROR] {e}: Insert file to cache failed, file cache database does not exist or might be in use!')


# def insert_zip_to_cache(db_path, arg1, arg2, arg3, arg4, arg5, arg6):
#     try:
#         cmd = "insert into file_cache(file_name, path, type, is_archive, n_content, container_archive) values ('"+arg1+"','"+arg2+"','"+arg3+"','"+arg4+"','"+arg5+"','"+arg6+"')"
#         #cmd = "insert into file_cache(file_name, path, type, is_archive, n_content, container_archive) values ('"+arg1+"', '"+arg2+"', '"+arg3+"', '"+arg4+"', '"+arg5+"', '"+arg6+"')"
#         with sqlite3.connect(db_path) as conn:
#             conn.execute(cmd)
#             conn.commit()
#     except Exception as e:
#         print (f'[ERROR] {e}: Insert file to cache failed, file cache database does not exist or might be in use!')


# @app.route('/clear_file_cache')
# def clear_file_cache():
    
#     try:

#         # Default
#         delete_cache= False
        
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')
#         _delete_cache = request.args.get('delete_cache')

#         print (type(_delete_cache))

#         if _db_path:
#             db_path = str(_db_path)
#         else:
#             raise Exception ('Database path not specified')

#         if _delete_cache:
#             if _delete_cache == "True":
#                 delete_cache = True
#             elif _delete_cache == "False":
#                 delete_cache = False
#             else:
#                 raise Exception ('Invalid argument')

#         if delete_cache:
#             # Delete file
#             os.remove(db_path)
#             return (f'[OK] File-cache database {db_path} has been removed.')
#         else:
#             # Clear table only
#             cmd = "DELETE FROM file_cache"
#             with sqlite3.connect(db_path) as conn:
#                 conn.execute(cmd)
#                 conn.commit()
#             return (f'[OK] Table "file_cache" on {db_path} database has been cleared.')
#     except Exception as e:
#         return (f'[ERROR] {e}: Clearing data from database failed!')


# @app.route('/get_random_file_hash')
# def get_random_file_hash():

#     try:
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        

#         if _db_path:
#             db_path = _db_path
#         else:
#             raise Exception ('Database path not specified')

#         # Getting random hash
#         cmd = "SELECT hash_id FROM file_cache ORDER BY RANDOM() LIMIT 1 ;"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return (json.dumps({'hash_id':row[0]}))

#     except Exception as e:
#         return (f'[ERROR]: {e} Getting random hash from cache failed!')


# @app.route('/get_img_by_hash')
# def get_img_by_hash():

#     try:

#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        
#         _hash_id = request.args.get('hash_id')

#         if _db_path and _hash_id:
#             db_path = _db_path
#             hash_id = str(_hash_id)
#         else:
#             raise Exception ('Database path or hash not specified')

#         cmd = "SELECT * FROM file_cache WHERE hash_id = '"+hash_id+"'"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return json.dumps({
#                         'file_name':row[0],
#                         'file_path':row[1],
#                         'hash_id':row[2],
#                         'file_type':row[3],
#                         'is_archive':row[4],
#                         'n_content':row[5],
#                         'container_archive':row[6],
#                         })

#     except Exception as e:
#         return (f'[ERROR] {e}: Getting image data from cache failed')


# @app.route('/get_random_file')
# def get_random_file():

#     try:

#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        

#         if _db_path:
#             db_path = _db_path
#         else:
#             raise Exception ('Database path not specified')

#         cmd = "SELECT * FROM file_cache ORDER BY RANDOM() LIMIT 1 ;"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return json.dumps({
#                         'file_name':row[0],
#                         'file_path':row[1],
#                         'hash_id':row[2],
#                         'file_type':row[3],
#                         'is_archive':row[4],
#                         'n_content':row[5],
#                         'container_archive':row[6],
#                         })
    
#     except Exception as e:
#         return (f'[ERROR] {e}: Getting image data failed, file cache database does not exist or might be in use!')


# def data_gen_for_file_cache(data_file):
#     '''Image generator for zip file'''
    
#     # Setting the placeholder
#     global zips_info
#     zips_info.clear()

#     if data_file.endswith('.zip'):
#         # Selected data_dir is a zip archive

#         print (f'[INFO] Processing ZIP archive: {data_file}')
        
#         with ZipFile(data_file) as archive:

#             '''Getting archive details'''
#             # Check the number of content (image file)
#             entries = archive.infolist()
#             n_content =  len([content for content in entries if content.is_dir() ==False])
#             # Appending to the list to be writen to database table
#             zips_info.append([data_file, n_content])

#             for entry in entries:
#                 # Do for every content in the zip file
#                 if not entry.is_dir():
                    
#                     with archive.open(entry) as file:

#                         if entry.filename.lower().endswith(('.zip')):
#                             # Another zip file found in the content.
#                             print (f'[INFO] Processing ZIP archive: {data_file}/{entry.filename}')
#                             # Process the content of the zip file
#                             with ZipFile(file) as sub_archive:

#                                 '''Getting archive details'''
#                                 # Check the number of content
#                                 sub_entries = sub_archive.infolist()
#                                 n_content =  len([content for content in sub_entries if content.is_dir() ==False])
#                                 # Appending to the list to be writen to database table
#                                 zips_info.append([f'{data_file}/{entry.filename}', n_content])

#                                 for sub_entry in sub_entries:
#                                     with sub_archive.open(sub_entry) as sub_file:
#                                         try:
#                                             img = Image.open(sub_file)
#                                             img_file_name = f'{data_file}/{entry.filename}/{sub_entry.filename}'
#                                             print (f' Fetching: {img_file_name}')
#                                             yield (img, img_file_name)
#                                         except:
#                                             print (f'[WWARNING] Failed to fetch {os.path.join(data_file, entry.filename, sub_entry.filename)}')
#                                             continue
#                         else:
#                             # Should be image file. Read it.
#                             try:
#                                 img = Image.open(file)
#                                 img_file_name = f'{data_file}/{entry.filename}'
#                                 print (f' Fetching: {img_file_name}')
#                                 yield (img, img_file_name)
#                             except:
#                                 print (f'[WARNING] Failed to fetch {data_file}/{entry.filename}')
#                                 continue
#     else:
#         # Should be image file. Read it.
#         try:
#             img = Image.open(data_file)
#             print (f' Fetching: {data_file}')
#             yield (img, data_file)
#         except:
#             print (f'[WARNING] Failed to fetch {data_file}')
    

# def compute_hash(img, img_file_name):
#     '''Compute image file to hash'''
#     try:
#         return hashlib.blake2b(img.tobytes()).hexdigest()
#     except Exception as e:
#         print(f"[ERROR]  cannot compute hash for {img_file_name} , {e}")
#         return None 

# def empty_dirs_check(dir_path):
#     """ Checking for empty directory and print out warning if any"""
#     for dir in os.listdir(dir_path):
#         sub_dir = os.path.join(dir_path, dir)
#         # Check for directory only
#         if os.path.isdir(sub_dir):
#             if len(os.listdir(sub_dir)) == 0:
#                 # Empty folder
#                 print(f'[WARNING] Empty folder found. Ignoring it: {sub_dir}')
#                 continue

# @app.route('/add_folder_to_file_cache')
# def add_folder_to_file_cache():
#     '''Load images to file-cache'''

#     # Default
#     out_dir= './output'
#     db_name = 'file_cache.sqlite'

#     try:
#         # Argument extraction from URL query string
#         _data_dir = request.args.get('data_dir')    
#         _out_dir = request.args.get('out_dir')
#         _db_name = request.args.get('db_name')

#         if _data_dir:
#             data_dir = _data_dir
#         else:
#             raise Exception ('Input directory not specified')
#         if _out_dir:
#             out_dir = _out_dir
#         if _db_name:
#             db_name = _db_name
    
#         db_path = f'{out_dir}/{db_name}'

#         if (os.path.exists(db_path)):
#             # Placeholder for data file names
#             files_list = []
            
#             if not os.path.isfile(data_dir):
#                 '''For normal directory'''
#                 # Check for empty dirs
#                 empty_dirs_check(data_dir)
#                 # Walking thru files
#                 for root, _, files in os.walk(data_dir):
#                     for file in files:
#                         files_list.append(os.path.join(root, file))
#             else:
#                 '''Single file (could be a zip archive or image)'''
#                 files_list = [data_dir]

#             for file in files_list:
#                 '''Fetching images'''
                
#                 # Check if it is container archive
#                 container_archive = ''
#                 if file.lower().endswith(('.zip')):
#                     container_archive = file

#                 for img, file_path in data_gen_for_file_cache(file):
#                     file_name = os.path.basename(file_path)
#                     type = os.path.splitext(file_name)[-1]
#                     #tag_folder = os.path.split(os.path.splitext(os.path.split(file_path)[0])[0])[-1]
#                     # Compute hash
#                     hash = compute_hash(img, file_path)  
#                     # Insert image to cache
#                     insert_file_to_cache(db_path, file_name, file_path, hash, type, container_archive)  

#             # Insert zip files to cache
#             for _zip in zips_info:
#                 parent = os.path.split(_zip[0])[0]
#                 if parent.lower().endswith(('.zip')):
#                     container_archive = parent
#                 else:
#                     pass
#                     #parent_folder = parent
#                 file_name = os.path.basename(_zip[0])
#                 arch_path = _zip[0]
#                 n_content = _zip[1]
#                 insert_zip_to_cache(db_path, file_name, arch_path, os.path.splitext(file_name)[-1], str(True), str(n_content), container_archive)

#         else:
#             return (f'[ERROR] Database {db_path} does not exist !')

#         return (f'[OK]: Folder {data_dir} added to {db_path}')
    
#     except Exception as e:
#         return (f'[ERROR] {e}: Adding data to database failed!')



# '''CLIP Cache'''

# @app.route('/create_clip_cache')
# def create_clip_cache():
#     '''
#     Creating clip-cache database
#     '''
#     # Default
    
#     out_dir = './output'
#     db_name = 'clip_cache.sqlite'

#     # Argument extraction from URL query string
#     _out_dir = request.args.get('out_dir')
#     _db_name = request.args.get('db_name')

#     if _out_dir:
#         out_dir = str(_out_dir)
#     if _db_name:
#         db_name = str(_db_name)

#     db_path = f'{out_dir}/{db_name}'

#     def __create_database(db_path):
#         cmd = '''CREATE TABLE clip_cache (
#         hash_id   TEXT          ,
#         clip_vector   BLOB      ,
#         model   TEXT      
#         );
#         '''
#         with sqlite3.connect(db_path) as conn:
#             conn.execute('PRAGMA encoding="UTF-8";')
#             conn.execute(cmd)
#             conn.commit()

#     try:
#         # Make sure result output path exists 
#         os.makedirs(out_dir, exist_ok = True) 
#         # Check for existing file
#         if(os.path.exists(db_path)):
#             return (f'[ERROR]: Previous {db_name} already exist in {db_path}. Clip-cache creation stopped.')
#         else:
#             __create_database(db_path)
#             return (f'[OK]: database {db_path} created')

#     except Exception as e:
#         return (f'[ERROR] {e}: Creating cache database failed!')

# def insert_clip_to_cache(db_path, arg1, arg2, arg3):
#     try:
#         cmd = """insert into clip_cache(hash_id, clip_vector, model) values (?, ?, ?)"""
#         with sqlite3.connect(db_path) as conn:
#             conn.execute(cmd, (arg1, arg2, arg3))
#             conn.commit()
#     except Exception as e:
#         print (f'[ERROR] {e}: Insert clip to cache failed, clip cache database does not exist or might be in use!')


# @app.route('/clear_clip_cache')
# def clear_clip_cache():
    
#     try:

#         # Default
#         delete_cache= False
        
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')
#         _delete_cache = request.args.get('delete_cache')

#         if _db_path:
#             db_path = str(_db_path)
#         else:
#             raise Exception ('Database path not specified')

#         if _delete_cache:
#             if _delete_cache == "True":
#                 delete_cache = True
#             elif _delete_cache == "False":
#                 delete_cache = False
#             else:
#                 raise Exception ('Invalid argument')

#         if delete_cache:
#             # Delete file
#             os.remove(db_path)
#             return (f'[OK] Clip cache database {db_path} has been removed.')
#         else:
#             # Clear table only
#             cmd = "DELETE FROM clip_cache"
#             with sqlite3.connect(db_path) as conn:
#                 conn.execute(cmd)
#                 conn.commit()
#             return (f'[OK] Table "clip_cache" on {db_path} database has been cleared.')
#     except Exception as e:
#         return (f'[ERROR] {e}: Clearing data from database failed!')


# @app.route('/get_random_clip_hash')
# def get_random_clip_hash():

#     try:
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        

#         if _db_path:
#             db_path = _db_path
#         else:
#             raise Exception ('Database path not specified')

#         # Getting random hash
#         cmd = "SELECT hash_id FROM clip_cache ORDER BY RANDOM() LIMIT 1 ;"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return (json.dumps({'hash_id':row[0]}))

#     except Exception as e:
#         return (f'[ERROR]: {e} Getting random hash from cache failed!')


# @app.route('/get_clip_by_hash')
# def get_clip_by_hash():

#     try:

#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        
#         _hash_id = request.args.get('hash_id')

#         if _db_path and _hash_id:
#             db_path = _db_path
#             hash_id = str(_hash_id)
#         else:
#             raise Exception ('Database path or hash not specified')

#         cmd = "SELECT * FROM clip_cache WHERE hash_id = '"+hash_id+"'"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return json.dumps({
#                         'hash_id':row[0],
#                         'clip_vector': base64.b64encode(row[1]).decode('ascii'),
#                         'model' : row[2]
#                         })

#     except Exception as e:
#         return (f'[ERROR] {e}: Getting image data from cache failed')


# @app.route('/get_random_clip')
# def get_random_clip():

#     try:

#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        

#         if _db_path:
#             db_path = _db_path
#         else:
#             raise Exception ('Database path not specified')

#         cmd = "SELECT * FROM clip_cache ORDER BY RANDOM() LIMIT 1 ;"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return json.dumps({
#                         'hash_id':row[0],
#                         'clip_vector': base64.b64encode(row[1]).decode('ascii'),
#                         'model' : row[2]
#                         })
    
#     except Exception as e:
#         return (f'[ERROR] {e}: Getting clip vector failed, clip cache database does not exist or might be in use!')


# def data_gen_for_clip_cache(data_file):
#     '''Generator for data_file'''
    
#     if data_file.endswith('.zip'):
#         # Selected data_dir is a zip archive
        
#         with ZipFile(data_file) as archive:

#             # Getting archive details
#             entries = archive.infolist()

#             for entry in entries:
#                 # Do for every content in the zip file
#                 if not entry.is_dir():
                    
#                     with archive.open(entry) as file:

#                         if entry.filename.lower().endswith(('.zip')):
#                             # Another zip file found in the content. Process the content of the zip file
#                             with ZipFile(file) as sub_archive:

#                                 '''Getting archive details'''
#                                 # Check the number of content
#                                 sub_entries = sub_archive.infolist()

#                                 for sub_entry in sub_entries:
#                                     with sub_archive.open(sub_entry) as sub_file:
#                                         try:
#                                             img = Image.open(sub_file)
#                                             yield (img)
#                                         except:
#                                             print (f'[WWARNING] Failed to fetch {os.path.join(data_file, sub_entry.filename)}')
#                                             continue
#                         else:
#                             # Should be image file. Read it.
#                             try:
#                                 img = Image.open(file)
#                                 yield (img)
#                             except:
#                                 print (f'[WARNING] Failed to fetch {entry.filename}')
#                                 continue
#     else:
#         # Should be image file. Read it.
#         try:
#             img = Image.open(data_file)
#             print (f' Fetching: {data_file}')
#             yield (img)
#         except:
#             print (f'[WARNING] Failed to fetch {data_file}')


# def get_clip(clip_model_type : str = 'ViT-B-32', pretrained : str = 'openai'):
#     # get clip model from open_clip
#     clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_type,pretrained=pretrained)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     return clip_model, preprocess, device


# def get_clip_vector(img, image_file_name : str, model, preprocess, device:str):
#     with torch.no_grad():
        
#         if image_file_name.lower().endswith('.gif'): 
#             try:
#                 img.seek(0)
#             except:
#                 print [f'[WARNING] Failed to convert {image_file_name} image.']
#                 return
#         else:
#             # Image files other than gif
#             img_obj = img

#         image = preprocess(img_obj).unsqueeze(0).to(device)
#         return model.encode_image(image).detach().numpy()


# @app.route('/add_folder_to_clip_cache')
# def add_folder_to_clip_cache():
#     '''Load images to clip-cache'''

#     # Default
#     out_dir= './output'
#     db_name = 'clip_cache.sqlite'
#     clip_model_type = 'ViT-B-32'
#     clip_model_pretrained = 'openai'

#     try:
#         # Argument extraction from URL query string
#         _data_dir = request.args.get('data_dir')    
#         _out_dir = request.args.get('out_dir')
#         _db_name = request.args.get('db_name')

#         if _data_dir:
#             data_dir = _data_dir
#         else:
#             raise Exception ('Input directory not specified')
#         if _out_dir:
#             out_dir = _out_dir
#         if _db_name:
#             db_name = _db_name
    
#         db_path = f'{out_dir}/{db_name}'

#         # Getting clip model
#         clip_model , preprocess , device = get_clip(clip_model_type= clip_model_type, pretrained= clip_model_pretrained)

#         if (os.path.exists(db_path)):

#             # Placeholder for data file names
#             files_list = []
            
#             if not os.path.isfile(data_dir):
#                 '''For normal directory'''
#                 # Check for empty dirs
#                 empty_dirs_check(data_dir)
#                 # Walking thru files
#                 for root, _, files in os.walk(data_dir):
#                     for file in files:
#                         files_list.append(os.path.join(root, file))
#             else:
#                 '''Single file (could be a zip archive or image)'''
#                 files_list = [data_dir]
        
#             for file in files_list:
#                 '''Fetching images'''
            
#                 for img in data_gen_for_clip_cache(file):
#                     # Compute hash
#                     hash = compute_hash(img, file)  
#                     print (f'[INFO] Calculating CLIP vector for {file}...')
#                     # Compute clip vector
#                     clip_vector = get_clip_vector(img, file, clip_model,preprocess,device)
#                     clip_vector = pickle.dumps(clip_vector)
#                     # Insert image to cache
#                     insert_clip_to_cache(db_path, hash, clip_vector, f'{clip_model_type}:{clip_model_pretrained}')  
#         else:
#             return (f'[ERROR] Database {db_path} does not exist !')

#         return (f'[OK]: Folder {data_dir} added to {db_path}')
    
#     except Exception as e:
#         return (f'[ERROR] {e}: Adding data to database failed!')


# '''TAG Cache'''

# @app.route('/create_tag_cache')
# def create_tag_cache():
#     '''
#     Creating tag-cache database
#     '''
#     # Default
    
#     out_dir = './output'
#     db_name = 'tag_cache.sqlite'

#     # Argument extraction from URL query string
#     _out_dir = request.args.get('out_dir')
#     _db_name = request.args.get('db_name')

#     if _out_dir:
#         out_dir = str(_out_dir)
#     if _db_name:
#         db_name = str(_db_name)

#     db_path = f'{out_dir}/{db_name}'

#     def __create_database(db_path):
#         cmd = '''CREATE TABLE tag_cache (
#         hash_id     TEXT,
#         tag     TEXT            
#         );
#         '''
#         with sqlite3.connect(db_path) as conn:
#             conn.execute('PRAGMA encoding="UTF-8";')
#             conn.execute(cmd)
#             conn.commit()

#     try:
#         # Make sure result output path exists 
#         os.makedirs(out_dir, exist_ok = True) 
#         # Check for existing file
#         if(os.path.exists(db_path)):
#             return (f'[ERROR]: Previous {db_name} already exist in {db_path}. Tag-cache creation stopped.')
#         else:
#             __create_database(db_path)
#             return (f'[OK]: database {db_path} created')

#     except Exception as e:
#         return (f'[ERROR] {e}: Creating cache database failed!')


# def insert_tag_to_cache(db_path, arg1, arg2):
#     try:
#         cmd = "insert into tag_cache(hash_id, tag) values ('"+arg1+"', '"+arg2+"')"
#         with sqlite3.connect(db_path) as conn:
#             conn.execute(cmd)
#             conn.commit()
#     except Exception as e:
#         print (f'[ERROR] {e}: Insert file to cache failed, file cache database does not exist or might be in use!')


# @app.route('/clear_tag_cache')
# def clear_tag_cache():
    
#     try:

#         # Default
#         delete_cache= False
        
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')
#         _delete_cache = request.args.get('delete_cache')

#         if _db_path:
#             db_path = str(_db_path)
#         else:
#             raise Exception ('Database path not specified')

#         if _delete_cache:
#             if _delete_cache == "True":
#                 delete_cache = True
#             elif _delete_cache == "False":
#                 delete_cache = False
#             else:
#                 raise Exception ('Invalid argument')

#         if delete_cache:
#             # Delete file
#             os.remove(db_path)
#             return (f'[OK] Tag cache database {db_path} has been removed.')
#         else:
#             # Clear table only
#             cmd = "DELETE FROM tag_cache"
#             with sqlite3.connect(db_path) as conn:
#                 conn.execute(cmd)
#                 conn.commit()
#             return (f'[OK] Table "tag_cache" on {db_path} database has been cleared.')
#     except Exception as e:
#         return (f'[ERROR] {e}: Clearing data from database failed!')


# @app.route('/get_random_tag_hash')
# def get_random_tag_hash():

#     try:
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        

#         if _db_path:
#             db_path = _db_path
#         else:
#             raise Exception ('Database path not specified')

#         # Getting random hash
#         cmd = "SELECT hash_id FROM tag_cache ORDER BY RANDOM() LIMIT 1 ;"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return (json.dumps({'hash_id':row[0]}))

#     except Exception as e:
#         return (f'[ERROR]: {e} Getting random hash from cache failed!')


# @app.route('/get_tag_by_hash')
# def get_tag_by_hash():

#     try:
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        
#         _hash_id = request.args.get('hash_id')

#         if _db_path and _hash_id:
#             db_path = _db_path
#             hash_id = str(_hash_id)
#         else:
#             raise Exception ('Database path or hash not specified')

#         cmd = "SELECT * FROM tag_cache WHERE hash_id = '"+hash_id+"'"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return (json.dumps({
#                         'hash_id':row[0],
#                         'tag':row[1],
#                         }))

#     except Exception as e:
#         return (f'[ERROR] {e}: Getting image data from cache failed')


# @app.route('/get_random_tag')
# def get_random_tag():

#     try:
#         # Argument extraction from URL query string
#         _db_path = request.args.get('db_path')        

#         if _db_path:
#             db_path = _db_path
#         else:
#             raise Exception ('Database path not specified')

#         cmd = "SELECT * FROM tag_cache ORDER BY RANDOM() LIMIT 1 ;"
#         with sqlite3.connect(db_path) as conn:
#             cur = conn.cursor()
#             cur.execute(cmd)
#             for row in cur:
#                 return json.dumps({
#                         'hash_id':row[0],
#                         'tag':row[1],
#                         })
    
#     except Exception as e:
#         return (f'[ERROR] {e}: Getting clip vector failed, clip cache database does not exist or might be in use!')


# def data_gen_for_tag_cache(data_file):
#     '''Image generator'''
    
#     if data_file.endswith('.zip'):
#         # Selected data_dir is a zip archive
        
#         with ZipFile(data_file) as archive:

#             '''Getting archive details'''
#             # Check the number of content (image file)
#             entries = archive.infolist()

#             for entry in entries:
#                 # Do for every content in the zip file
#                 if not entry.is_dir():
                    
#                     with archive.open(entry) as file:

#                         if entry.filename.lower().endswith(('.zip')):
#                             # Another zip file found in the content.
#                             print (f'[INFO] Processing ZIP archive: {data_file}/{entry.filename}')
#                             # Process the content of the zip file
#                             with ZipFile(file) as sub_archive:

#                                 '''Getting archive details'''
#                                 sub_entries = sub_archive.infolist()

#                                 for sub_entry in sub_entries:
#                                     with sub_archive.open(sub_entry) as sub_file:
#                                         try:
#                                             img = Image.open(sub_file)
#                                             img_file_name = f'{data_file}/{entry.filename}/{sub_entry.filename}'
#                                             print (f' Processing: {img_file_name}')
#                                             yield (img, img_file_name)
#                                         except:
#                                             print (f'[WARNING] Failed to process {os.path.join(data_file, entry.filename, sub_entry.filename)}')
#                                             continue
#                         else:
#                             # Should be image file. Read it.
#                             try:
#                                 img = Image.open(file)
#                                 img_file_name = f'{data_file}/{entry.filename}'
#                                 print (f' Processing: {img_file_name}')
#                                 yield (img, img_file_name)
#                             except:
#                                 print (f'[WARNING] Failed to process {data_file}/{entry.filename}')
#                                 continue
#     else:
#         # Should be image file. Read it.
#         try:
#             img = Image.open(data_file)
#             print (f' Processing: {data_file}')
#             yield (img, data_file)
#         except:
#             print (f'[WARNING] Failed to process {data_file}')


# @app.route('/add_folder_to_tag_cache')
# def add_folder_to_tag_cache():
#     '''Load images to tag-cache'''

#     # Default
#     out_dir= './output'
#     db_name = 'tag_cache.sqlite'

#     try:
#         # Argument extraction from URL query string
#         _data_dir = request.args.get('data_dir')    
#         _out_dir = request.args.get('out_dir')
#         _db_name = request.args.get('db_name')

#         if _data_dir:
#             data_dir = _data_dir
#         else:
#             raise Exception ('Input directory not specified')
#         if _out_dir:
#             out_dir = _out_dir
#         if _db_name:
#             db_name = _db_name
    
#         db_path = f'{out_dir}/{db_name}'

#         if (os.path.exists(db_path)):

#             # Placeholder for data file names
#             files_list = []
            
#             if not os.path.isfile(data_dir):
#                 '''For normal directory'''
#                 # Check for empty dirs
#                 empty_dirs_check(data_dir)
#                 # Walking thru files
#                 for root, _, files in os.walk(data_dir):
#                     for file in files:
#                         files_list.append(os.path.join(root, file))
#             else:
#                 '''Single file (could be a zip archive or image)'''
#                 files_list = [data_dir]
        
#             for file in files_list:
#                 '''Fetching images'''
            
#                 for img, file_path in data_gen_for_tag_cache(file):
#                     # Consider only image files within tag folders. Its root != data_dir 
#                     if (os.path.split(file_path)[0].strip() != data_dir.strip()):
#                         tag_folder = os.path.split(os.path.splitext(os.path.split(file_path)[0])[0])[-1]
#                         # Compute hash
#                         hash = compute_hash(img, file_path)  
#                         # Insert image to cache
#                         insert_tag_to_cache(db_path, hash, tag_folder)  
#                     else:
#                         print(f'[WARNING] : File {file_path} is outside of tag folder. Ignoring it...')
        
#         else:
#             return (f'[ERROR] Database {db_path} does not exist !')

#         return (f'[OK]: Folder {data_dir} added to {db_path}')
    
#     except Exception as e:
#         return (f'[ERROR] {e}: Adding data to database failed!')


if __name__=='__main__':

 # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=False, default='0.0.0.0')
    parser.add_argument('--port', type=int, required=False , default=8080)

    args = parser.parse_args()

    HOST=args.host
    PORT_NUMBER=args.port

    app.run(host=HOST, port=PORT_NUMBER)

