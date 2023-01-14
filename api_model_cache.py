import os 
import sqlite3
import json


class ModelCache(object):

    db_path = './output/stage3.sqlite'
    db_table_name = 'stage1'

    def __init__(self) -> None:
        pass

    def get_img_from_model_cache(self, model_name, score_gte=0.0, score_lte=1.0, db_path='./output/stage3.sqlite', db_table_name='stage3'):
        '''Returns list of file names for specific model_name and score score_gte <= score <= score_lte'''

        files_dict={}

        try:
            cmd = f"SELECT * FROM {db_table_name} WHERE tag_prob BETWEEN {score_gte} AND {score_lte}" #"' AND model_name '"+model_name+"'"
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(cmd)
                for row in cur:
                    hash_id = row[3]
                    files_dict[hash_id] = {
                                    'file_name':row[0],
                                    'file_path':row[1],
                                    'file_type':row[2],
                                    'model_name':row[4],
                                    'model_type':row[5],
                                    'tag':row[6],
                                    'score':row[7]
                                    }

            print (json.dumps(files_dict, indent = 2))
            return True, files_dict

        except Exception as e:
            print (f'[ERROR] {e}: Failed getting image data')
            return False, files_dict