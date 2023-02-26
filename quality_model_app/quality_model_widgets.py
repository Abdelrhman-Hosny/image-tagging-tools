import os
import random
import hashlib
import uuid
import json
import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output


class QualityModelWidgets(object):

    # Current Step
    n = 1
    # Placeholder for currently displayed images
    file_dict_1 = {}
    file_dict_2 = {}

    def __init__(self, data_dict, output_path = 'output.json') -> None:
        self.data_dict = data_dict
        self.output_path = output_path

    def start(self):

        # Initial Images
        self.file_dict_1, self.file_dict_2 = self.get_2_rand_images(self.data_dict)
        self.img_widget_1 = widgets.Image(value=self.file_dict_1['img_bytes'], format='jpg', width=300, height=400)
        self.img_widget_2 = widgets.Image(value=self.file_dict_2['img_bytes'], format='jpg', width=300, height=400)
        # Title label
        self.lbl_title_value = f'Quality Model App'
        self.lbl_title = widgets.HTML(value=f'<p style="font-size: 24px ; font-weight: bold ; color:rgb(75,75,75)">{self.lbl_title_value}</p>')
        # Tagging User
        self.lbl_user = widgets.HTML(value=f'<p style="font-size: 16px ; font-weight: bold ; color:rgb(75,75,75) ; height: 20px">Tagging User: </p>')
        self.txt_user = widgets.Text(value='', disabled=False)
        self.txt_user.layout.width = '250px'
        # Task String User
        self.lbl_task = widgets.HTML(value=f'<p style="font-size: 16px ; font-weight: bold ; color:rgb(75,75,75) ; height: 20px">Task String: </p>')
        self.txt_task = widgets.Text(value='', disabled=False)
        self.txt_task.layout.width = '250px'
        # Status label
        lbl_status_value = f'Choose Best Image - {self.n}'
        self.lbl_status = widgets.HTML(value=f'<p style="font-size: 20px ; font-weight: bold ; color:rgb(75,75,75)">{lbl_status_value}</p>')
        # Selection buttons
        self.btn_select_1 = widgets.Button(description = 'SELECT', icon='check', button_style = 'success')
        self.btn_select_1.style.button_color = 'rgb(30,144,255)'
        self.btn_select_2 = widgets.Button(description = 'SELECT', icon='check', button_style = 'success')
        self.btn_select_2.style.button_color = 'rgb(30,144,255)'
        # Skip button
        self.btn_skip = widgets.Button(description = 'SKIP')
        self.btn_skip.style.button_color = 'rgb(225,225,225)'
        # Layout
        self.box_layout = widgets.Layout(display='flex',
                                    flex_flow='row',
                                    justify_content = 'space-around',
                                    align_items='center',
                                    width='100%'
                                    )
        
        # binding skip button to skip function callback
        self.btn_skip.on_click(self.skip_pressed)
        # binding select button 1 and 2 to select function callback
        self.btn_select_1.on_click(self.select_pressed)
        self.btn_select_2.on_click(self.select_pressed)

        # Show widgets
        self.show_widgets(
                    self.lbl_title,
                    self.lbl_status,
                    self.lbl_user, 
                    self.txt_user, 
                    self.lbl_task, 
                    self.txt_task, 
                    self.img_widget_1, 
                    self.img_widget_2, 
                    self.btn_select_1, 
                    self.btn_select_2, 
                    self.btn_skip, 
                    self.box_layout
                    )


    def get_2_rand_images (self, data_dict):
        
        # List of hashes (keys in data_dict)
        hash_list = list(data_dict.keys())
        # File 1
        hash_1 = random.choice(hash_list)
        file_path_1 = data_dict[hash_1]['file_path']
        file_name_1 = data_dict[hash_1]['file_name']
        img_bytes_1 = data_dict[hash_1]['img_bytes']
        # File 2
        hash_2 = random.choice(hash_list)
        file_path_2 = data_dict[hash_2]['file_path']
        file_name_2 = data_dict[hash_2]['file_name']
        img_bytes_2 = data_dict[hash_2]['img_bytes']

        # # Image 1 bytes serialization
        # img_byte_1 = io.BytesIO()
        # img_1.save(img_byte_1, format = img_1.format)
        # img_byte_1.seek(0)
        # img_data_1 = img_byte_1.read()

        file_dict_1 = {'hash': hash_1, 'file_path': file_path_1, 'file_name': file_name_1, 'img_bytes': img_bytes_1}
        file_dict_2 = {'hash': hash_2, 'file_path': file_path_2, 'file_name': file_name_2, 'img_bytes': img_bytes_2}

        return file_dict_1, file_dict_2


    def show_widgets(self, lbl_title, lbl_status, lbl_user, txt_user, lbl_task, txt_task, img_1, img_2, btn_select_1, btn_select_2, btn_skip, box_layout):
        self.box_title = widgets.Box(children=[lbl_title], layout=widgets.Layout(display='flex', flex_flow='row', justify_content = 'flex-start', align_items='center', width='100%'))
        self.box_user = widgets.Box(children=[lbl_user, txt_user], layout=widgets.Layout(display='flex', flex_flow='row', justify_content = 'flex-start', align_items='center', width='100%'))
        self.box_task = widgets.Box(children=[lbl_task, txt_task], layout=widgets.Layout(display='flex', flex_flow='row', justify_content = 'flex-start', align_items='center', width='100%'))
        self.box_user_task = widgets.Box(children=[self.box_user, self.box_task], layout=widgets.Layout(display='flex', flex_flow='row', justify_content = 'flex-start', align_items='center', width='100%'))
        self.box_status = widgets.Box(children=[lbl_status], layout=widgets.Layout(display='flex', flex_flow='row', justify_content = 'flex-start', align_items='center', width='100%'))
        self.box_images = widgets.Box(children=[img_1, img_2], layout=box_layout)
        self.box_select = widgets.Box(children=[btn_select_1, btn_select_2], layout=box_layout)
        self.box_skip = widgets.Box(children=[btn_skip], layout=box_layout)
        display(self.box_title)
        display(self.box_user_task)
        display(self.box_status)
        display(self.box_images)
        display(self.box_select)
        display(self.box_skip)


    def skip_pressed(self, button):
        # Currently displayed images
        self.file_dict_1
        self.file_dict_2
        # Increment step
        self.n += 1
        clear_output()
        # Update status label
        lbl_status_value = f'Choose Best Image - {self.n}'
        self.lbl_status = widgets.HTML(value=f'<p style="font-size: 20px ; font-weight: bold ; color:rgb(75,75,75)">{lbl_status_value}</p>')
        # Get new images
        self.file_dict_1, self.file_dict_2 = self.get_2_rand_images(self.data_dict)
        self.img_widget_1 = widgets.Image(value=self.file_dict_1['img_bytes'], format='jpg', width=300, height=400)
        self.img_widget_2 = widgets.Image(value=self.file_dict_2['img_bytes'], format='jpg', width=300, height=400)
        self.show_widgets(
                    self.lbl_title,
                    self.lbl_status,
                    self.lbl_user, 
                    self.txt_user, 
                    self.lbl_task, 
                    self.txt_task, 
                    self.img_widget_1, 
                    self.img_widget_2, 
                    self.btn_select_1, 
                    self.btn_select_2, 
                    self.btn_skip, 
                    self.box_layout
                    )
        

    def select_pressed(self, button):

        # Increment step
        self.n += 1

        # Time Stamp
        timestamp = datetime.datetime.now() 
        timestamp_str = f'{timestamp.year}_{timestamp.month}_{timestamp.day}_{timestamp.hour}_{timestamp.minute}'
        
        '''Which image is selected'''
        if button == self.btn_select_1:
            # Image 1 is selected
            self.save_to_json_file(selected = self.file_dict_1, options = [self.file_dict_1, self.file_dict_2], time_stamp = timestamp_str, output_path = self.output_path)
        elif button == self.btn_select_2:
            # Image 2 is selected
            self.save_to_json_file(selected = self.file_dict_2, options = [self.file_dict_1, self.file_dict_2], time_stamp = timestamp_str, output_path = self.output_path)

        # Clearing widgets
        clear_output()
        # Update status label
        lbl_status_value = f'Choose Best Image - {self.n}'
        self.lbl_status = widgets.HTML(value=f'<p style="font-size: 20px ; font-weight: bold ; color:rgb(75,75,75)">{lbl_status_value}</p>')
        # Get new images
        self.file_dict_1, self.file_dict_2 = self.get_2_rand_images(self.data_dict)
        self.img_widget_1 = widgets.Image(value=self.file_dict_1['img_bytes'], format='jpg', width=300, height=400)
        self.img_widget_2 = widgets.Image(value=self.file_dict_2['img_bytes'], format='jpg', width=300, height=400)
        self.show_widgets(
                    self.lbl_title,
                    self.lbl_status,
                    self.lbl_user, 
                    self.txt_user, 
                    self.lbl_task, 
                    self.txt_task, 
                    self.img_widget_1, 
                    self.img_widget_2, 
                    self.btn_select_1, 
                    self.btn_select_2, 
                    self.btn_skip, 
                    self.box_layout
                    )


    def save_to_json_file(self, selected, options, time_stamp, output_path):
        # Unique ID
        uid = str(uuid.uuid4())
        # JSON
        out_json = {'uid': uid, 'taskname': self.txt_task.value, 'input_image1': options[0]['hash'], 'input_image2': options[1]['hash'], 'selected_image':selected['hash'], 'user':self.txt_user.value, 'timestamp':time_stamp}
        # Serializing json
        json_object = json.dumps(out_json, indent=4)    
        # Writing to output folder
        with open(output_path, "a") as outfile:
            outfile.write(json_object)
            outfile.write('\n')



''' Data dictionay craetor function'''
def create_input_data_dict(input_dir):

    # Placeholder for dict to contain result from running on data source.
    data_dict = {}

    print ('[INFO] Running on Data Source...')

    # Walking thru files
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Get file path
            file_path = f'{root}/{file}'
            # Check if file is png or jpg
            if os.path.splitext(file_path)[-1] == '.png' or os.path.splitext(file_path)[-1] == '.jpg':
                try:
                    # Get image bytes
                    with open(file_path, 'rb') as img_file:
                        img_bytes = img_file.read()
                    # Compute hash
                    hasher = hashlib.sha256()
                    hasher.update(img_bytes)
                    hash_id = hasher.hexdigest()
                    data_dict[hash_id]={'file_path':file_path, 'file_name':file, 'img_bytes':img_bytes}
                except Exception as e:
                    print [f'[WARNING] Error when processing file: {e}']
            
    # Number of images
    n_images = len(data_dict)
    print (f'[INFO] Completed. Number of images: {n_images}')

    return data_dict
