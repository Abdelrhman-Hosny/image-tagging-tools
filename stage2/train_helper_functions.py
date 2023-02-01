import os
import numpy as np
from sklearn import metrics
from typing import List, Tuple, Union
from ascii_graph import Pyasciigraph
import joblib
import json
import torch


def make_dir(dir_names : Union[List[str] , str]):
    """takes a list of strings or a string and make a directory based on it.
    
    :param dir_name: the name(s) which will be the path to the directory.
    :type dir_name: Union[List[str] , str]
    :returns: a path to the new directory created 
    :rtype: str
    """
    if type(dir_names) == str:
        if dir_names.strip() == "":
            raise ValueError("Please enter a name to the directory")
   
        os.makedirs(dir_names , exist_ok=True)
        return dir_names
  
    elif type(dir_names) == list and len(dir_names) == 0:
        raise ValueError("Please enter list with names")
  
    elif type(dir_names) == list and len(dir_names) == 1:
        os.makedirs(dir_names[0] , exist_ok=True)
        return dir_names[0]
  
    final_dir = os.path.join(dir_names[0] , dir_names[1])
    for name in dir_names[2:]:
        final_dir = os.path.join(final_dir , name)

    os.makedirs(final_dir , exist_ok=True)
    return final_dir


def load_json(json_file_path:str):
    """ Takes a path for json file then returns a dictionary of it.

        :param json_file_path: path to the json file.
        :type json_file_path: str
        :returns: dictionary of the json file.
        :rtype: dict
    """
    if json_file_path != None:
      try :
        with open(json_file_path, 'rb') as json_obj:
          json_dict = json.load(json_obj)
        return json_dict
      
      except Exception as e : # handles any exception of the json file
        print(f"[ERROR] Problem loading {json_file_path}")
        return None
    else:
      return None


def get_train_test(
                    tag_all_emb_list: List, 
                    other_all_emb_list: List,
                    test_per: float
                    ):
    """takes embeding list of tag/class and other images,
    converts them into arrays ready for the training/testing.

    :param tag_all_emb_list: list of embeddings for the tag images.
    :type tag_all_emb_list: List
    :param other_all_emb_list: list of embeddings for the other images.
    :type other_all_emb_list: List
    :param test_per: percentage of the test images from the embeddings list.
    :type test_per: float
    :returns: tuple of the train_embds , train_labels , test_embs , test_labels \
              number of tags test images , number of other test images 
    :rtype: tuple
    """
    # get the test embeds from both classes (tag class and other)
    train_emb_list   = []
    train_label_list = []
    test_emb_list    = []
    test_label_list  = []

    # size of the number of the test set of the tag/class 
    tag_n_test = int(test_per*len(tag_all_emb_list)) if  int(test_per*len(tag_all_emb_list))  > 0 else 1 
    test_tag_emb_list    = tag_all_emb_list[: tag_n_test] # test tag/class embeddings. 
    test_emb_list.extend(test_tag_emb_list)
    test_tag_label_list  = [0] * len(test_tag_emb_list)   # test labels for tag/class embeddings
    test_label_list.extend(test_tag_label_list)
    train_tag_emb_list   = tag_all_emb_list[tag_n_test:] if len(tag_all_emb_list[tag_n_test:]) > 0 else tag_all_emb_list # train tag/class embeddings.
    train_emb_list.extend(train_tag_emb_list)
    train_tag_label_list = [0] * len(train_tag_emb_list)  # train labels for tag/class embeddings
    train_label_list.extend(train_tag_label_list)

    # size of the number of the test set of the tag/class 
    other_n_test = int(test_per*len(other_all_emb_list)) if  int(test_per*len(other_all_emb_list))  > 0 else 1 
    test_other_emb_list    = other_all_emb_list[:other_n_test]    # test other embeddings.
    test_emb_list.extend(test_other_emb_list)
    test_other_label_list  = [1] * len(test_other_emb_list)       # test labels for other embeddings.        
    test_label_list.extend(test_other_label_list) 
    train_other_emb_list   = other_all_emb_list[other_n_test:] if len(other_all_emb_list[other_n_test:]) > 0 else  other_all_emb_list   # train other embeddings.
    train_emb_list.extend(train_other_emb_list)
    train_tag_label_list   = [1] * len(train_other_emb_list)      # train labels for tag/class embeddings.
    train_label_list.extend(train_tag_label_list)

    # convert all of these lists into numpy arrays and returns them. 
    return np.array(train_emb_list), np.array(train_label_list), np.array(test_emb_list), np.array(test_label_list), \
           tag_n_test, other_n_test


def calc_confusion_matrix(
                          test_labels , 
                          predictions ,
                          tag_name : str 
                         ):
    """calculate accuracy, confusion matrix parts and return them.

    :param test_labels: labels for the test embeddings.
    :type test_labels: NdArray
    :param predictions: prediction from the classifer for the test_labels.
    :type predictions: NdArray
    :returns: accuracy,false positive rate, false negative rate, true positive rate, \
              true negative rate, false positive, false negative, true positive, true negative.
    :rtype: list of strings  
    """
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    ALL_SUM = FP + FN + TP + TN
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    return [ 
             f'false-positive-rate: {FPR[0] :.4f}  \n', 
             f'false-negative-rate: {FNR[0] :.4f}  \n',
             f'true-positive-rate : {TPR[0] :.4f}  \n',
             f'true-negative-rate : {TNR[0] :.4f}  \n\n',
             f'false-positive :  {FP[0]} out of {ALL_SUM[0]}  \n',
             f'false-negative : {FN[0]}  out of {ALL_SUM[0]} \n',
             f'true-positive : {TP[0]} out of {ALL_SUM[0]}  \n',
             f'true-negative : {TN[0]} out of {ALL_SUM[0]}  \n\n',
             f'>Accuracy : {accuracy:.4f}\n\n',
             f"Classification Report : \n\n{metrics.classification_report(test_labels, predictions)}\n\n",
             f"Index 0 is class {tag_name}\n",
             "Index 1 is class other \n\n"
            ]


def histogram_list(
                   image_features_list,
                   reg_model,
                   other = False,
                   using_torch=False,
                   ):
        """calculate the histogram for a group of images in features.

        :param image_features_list: list of images's features
        :type image_features_list: List
        :param reg_model: object of the regression model.
        :type reg_model: Object
        :param other: true if we want to find the histogram for others.
        :type other: Bool
        :returns: a list of tuples containg string of prob. and number of occurence.
        :rtype: List[Tuple] 
        """
        hist_dict = {   
                        '0.1' : 0 ,'0.2' : 0,
                        '0.3' : 0 ,'0.4':0 ,
                        '0.5' : 0 ,'0.6':0 ,
                        '0.7' : 0 ,'0.8':0 ,
                        '0.9' : 0 ,'1.0':0 ,
                     } # dictionary of histogram values 
  
        for image_features in image_features_list:
            if not using_torch:
                prob = reg_model.predict_proba(image_features.reshape(1,-1))[0][1] if other else reg_model.predict_proba(image_features.reshape(1,-1))[0][0]
            else:
                other_prob = reg_model(torch.from_numpy(image_features.reshape(1,-1).astype(np.float32))).detach().numpy()[0][0]
                tag_prob   = (1 - other_prob)
                prob = other_prob if other else tag_prob
         
            # Choosing the bin of the image for index 0 
            if 0.0 <= prob <= 0.1 : 
                hist_dict['0.1'] += 1
            elif 0.1 < prob <= 0.2 : 
                hist_dict['0.2'] += 1
            elif 0.3 < prob < 0.4 : 
                hist_dict['0.3'] += 1
            elif 0.4 <= prob < 0.5 : 
                hist_dict['0.4'] += 1
            elif 0.5 <= prob < 0.6 : 
                hist_dict['0.5'] += 1
            elif 0.6 <= prob < 0.7 : 
                hist_dict['0.6'] += 1
            elif 0.7 <= prob < 0.8 : 
                hist_dict['0.7'] += 1
            elif 0.8 <= prob < 0.9 : 
                hist_dict['0.8'] += 1
            elif 0.9 <= prob < 1.0 : 
                hist_dict['0.9'] += 1
            elif prob == 1.0 : 
                hist_dict['1.0'] += 1 
   
        return  [(key , hist_dict[key]) for key in hist_dict]


def histogram_lines(
                    data  : List[Tuple],
                    title : str
                    ):
    """takes histogram data and return a list of strings,
    to be used in furthur report generation.

    :param data: list of tuples containing the data of the histogram ('category' , no_of_occurence)
                 ex: [('0.1' , 5) , ('0.2' , 4)]
    :type data: List[Tuple]
    :param title: title string of the histogram.
    :type title: string
    :returns: list of strings used in report genreation.
    :rtype: List[str] 
    """
    text_file_lines = []
    graph = Pyasciigraph()

    text_file_lines = [f'{line} \n' for line in graph.graph(title, data)]
    text_file_lines.append('\n\n')

    return text_file_lines


def check_out_folder(output_dir : str):
    """take out folder path and returns the two paths
       for reports and models.

       :param output_dir: path to the output directory.
       :type output_dir: str
       :retunrs: tuple of the reports output folder and models output folder.
       :rtype: Tuple[str] 
    """

    if output_dir is None:
        report_out_folder = make_dir(['output' , 'reports']) 
        models_out_folder = make_dir(['output' , 'models'])
    else:
        report_out_folder = os.path.join(output_dir , 'reports') 
        models_out_folder = os.path.join(output_dir , 'models')
        os.makedirs(report_out_folder , exist_ok=True)
        os.makedirs(models_out_folder , exist_ok=True)
    
    return report_out_folder , models_out_folder


def generate_report(
                    reports_output_folder : str,
                    tag_name : str, 
                    text_file_lines : List[str], 
                    model_name: str,
                    ):
    """generate text file with text file lines provided, 
       save it in output directory.

       :param reports_output_folder: output folder for saving report file.
       :type reports_output_folder: str
       :param tag_name: name of the classifer tag.
       :type tag_name: str
       :param model_name: name of the model .
       :type  model_name: str
       :rtype: None. 
    """

    model_file_name = f'model-report-{model_name}-tag-{tag_name}'
    text_file_path = os.path.join(reports_output_folder ,f'{model_file_name}.txt' )
    with open( text_file_path ,"w+", encoding="utf-8") as f:
        f.writelines(text_file_lines)
    f.close()

    return

def generate_model_file(
                        models_output_folder : str, 
                        classifier,
                        model_type,
                        train_start_time, 
                        tag_name : str
                       ):
    """
    takes model's object and convert it into pickle file.

    :param models_output_folder: path to the output folder to put oickle files.
    :type  models_output_folder: str
    :param classifer: classifer object to be saved.
    :type classifer: Object
    :param tag_name: name of the tag of the classifer.
    :type tag_name: str
    :param model_name: name of the model .
    :type  model_name: str
    :rtype: None 
    """

    # save classifier object into pickle file. 
    model_file_name = f'model-{model_type}-tag-{tag_name}'
    pickle_file_path = os.path.join(models_output_folder , f'{model_file_name}.pkl')
    model_dict = {'classifier':classifier, 'model_type': model_type, 'train_start_time': train_start_time, 'tag': tag_name}
    print (f'[INFO] Creating model file: {model_file_name}')
    joblib.dump(model_dict , pickle_file_path)
    return 


# Logistic Regression Pytorch class.
# class LogisticRegressionPytorch(torch.nn.Module):
#      def __init__(self, input_dim, output_dim):
#          #super(LogisticRegressionPytorch, self).__init__()
#          self.linear = torch.nn.Linear(input_dim, output_dim)
#      def forward(self, x):
#          return torch.sigmoid(self.linear(x))

def train_loop(
                model,
                train_emb,
                train_labels,
                epochs: int = 20000, 
                ):
    """Taining loop for LogisticRegressionPytorch object.
    
    :param model: LogisticRegressionPytorch model object
    :type model: LogisticRegressionPytorch
    :param train_emb: embedding for the training features.
    :type train_emb: Numpy.Ndarray.
    :param train_labels: labels for training set.
    :type train_labels: Numpy.NdArray.
    :param epochs: number of epochs
    :type epochs: int
    :returns: model after training. 
    :rtype: LogisticRegressionPytorch
    """

    # Converting the dataset into Pytorch tensors.
    train_emb   =torch.from_numpy(train_emb.astype(np.float32))
    train_labels=torch.from_numpy(train_labels.astype(np.float32))
    train_labels=train_labels.view(train_labels.shape[0],1)

    criterion   = torch.nn.BCELoss()
    optimizer   = torch.optim.SGD(model.parameters(),lr=0.01)

    for epoch in range(epochs+1):
        
        y_prediction=model(train_emb)
        loss=criterion(y_prediction,train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # if (epoch+1)%10000 == 0:
        #     print('epoch:', epoch+1,',loss=',loss.item())
    
    return model


