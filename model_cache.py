import os 
import sqlite3
import json
from datetime import datetime
from api_model import ModelApi


class ModelCache(object):

    def __init__(self) -> None:
        pass

    def get_img_from_score_cache(self, model_name, score_gte=0.0, score_lte=1.0, db_path='./output/score_cache.sqlite', db_table_name='score_cache'):
        '''
        Returns list of file names for specific model_name and score score_gte <= score <= score_lte 
        from classification cache specified in db_path
        '''

        files_dict={}

        try:
            cmd = f"SELECT * FROM {db_table_name} WHERE (tag_score BETWEEN {score_gte} AND {score_lte})"+" AND (model_name= '"+model_name+"')"
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
                                    'model_train_date':row[6],
                                    'tag':row[7],
                                    'score':row[8]
                                    }

            print (json.dumps(files_dict, indent = 2))
            return True, files_dict

        except Exception as e:
            print (f'[ERROR] {e}: Failed getting image data')
            return False, files_dict
    

    def clear_score_cache_by_model_date(self, models_path='./output/models/', score_cache_path='./output/score_cache.sqlite', score_cache_table_name='score_cache'):
        '''
        Clearing score cache from entry with model training date older than current model training date.
        '''

        deleted_entries=[]
        n_row_original = 0
        n_row_final = 0
        
        try:
            # Getting models dict from model pickle files in given models_path
            model_api = ModelApi()
            models_dict = model_api.get_models_dict(models_path)
            
            if len(models_dict) > 0:

                 # Count the entries at original condition
                cmd =   f"SELECT * FROM {score_cache_table_name}"
                with sqlite3.connect(score_cache_path) as conn:
                    cur = conn.cursor()
                    cur.execute(cmd)
                    n_row_original = len(cur.fetchall())

                for model_name in models_dict:
                    # Do clearing for each model

                    # Getting date form current model
                    current_model_date_str = models_dict[model_name]['train_start_time'].strftime('%Y-%m-%d, %H:%M:%S')

                    cmd =   f"SELECT * FROM {score_cache_table_name} "+ "WHERE (model_name= '"+model_name+"') AND (model_train_date < '"+current_model_date_str+"')"
                    with sqlite3.connect(score_cache_path) as conn:
                        cur = conn.cursor()
                        cur.execute(cmd)
                        for row in cur:
                            hash_id = row[3]
                            deleted_entries.append(hash_id)
                            # deleted_entries[hash_id] = {
                            #                 'file_name':row[0],
                            #                 'file_path':row[1],
                            #                 'file_type':row[2],
                            #                 'model_name':row[4],
                            #                 'model_type':row[5],
                            #                 'model_train_date':row[6],
                            #                 'tag':row[7],
                            #                 'score':row[8]
                            #                 }

                    # Delete for record in database which model date is earlier than current_model_data
                    cmd =   f"DELETE FROM {score_cache_table_name} "+ "WHERE (model_name= '"+model_name+"') AND (model_train_date < '"+current_model_date_str+"')"
                    with sqlite3.connect(score_cache_path) as conn:
                        cur = conn.cursor()
                        cur.execute(cmd)
                
                # Count the entries after removal
                cmd =   f"SELECT * FROM {score_cache_table_name}"
                with sqlite3.connect(score_cache_path) as conn:
                    cur = conn.cursor()
                    cur.execute(cmd)
                    n_row_final = len(cur.fetchall())

                print (f'Original number of entries: {n_row_original}')
                print (f'Number of deleted entries: {len(deleted_entries)}')
                #print (f'Number of deleted entries: {len(deleted_entries.keys())}')
                print (f'Current number of entries: {n_row_final}')
                print ('Deleted Entries:')
                print (deleted_entries[:20])
                    #print (json.dumps(deleted_entries, indent = 2))

                return True, deleted_entries
            
            else:
                raise Exception(f'Failed getting models dictionary from {models_path}')
                
        except Exception as e:
            print (f'[ERROR]: {e}')
            return False