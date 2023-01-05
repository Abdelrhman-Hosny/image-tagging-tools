import os
import sqlite3
import io
from zipfile import ZipFile
from PIL import Image
from flask import Flask, request, render_template
import base64
import time


'''File Cache Web API'''

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
            request_time = f'{round((t2-t1), 3)} s'
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


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000)

