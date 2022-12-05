import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split as tts
import multiprocessing as mp
import matplotlib.pyplot as plt




class dataset():

    def __init__(self,root_dir,trainsplit=.3, testsplit=.7, h = 6, w = 5):
        self.train = []
        self.test = []
        s = 0
        self.h = h
        self.w = w
        self.ft = h * w
        self.root_dir = root_dir
        self.trainsplit = trainsplit
        self.testsplit = testsplit
        dirs = os.listdir(root_dir)
        dirs.remove(".DS_Store")
        self.dirs = dirs
        with mp.Pool(mp.cpu_count()) as pool:
            for pics in tqdm(pool.imap(self.loadImages, range(len(dirs))), total=len(self.dirs)):
                s += 1
                ttrain,ttest = tts(pics,test_size=testsplit,train_size=trainsplit)
                self.train.append(ttrain)
                self.test.append(ttest)



        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.nsamps = self.train.shape[1]
        self.train = self.train.reshape((-1, h * w + 1))
        self.test = self.test.reshape((-1,h * w + 1))
        np.random.shuffle(self.test)

    def loadImages(self, dirInd):
        dir = self.dirs[dirInd]
        subjectpics = []
        for f in os.listdir('ExtendedYaleB/'+dir+'/'):
            if(f[-3:] == 'pgm' and f[-5] != 't'):
                check = np.array(Image.open('ExtendedYaleB/' + dir + '/' + f).resize((self.h,self.w))).flatten().astype(float)
                check /= np.linalg.norm(check)
                r = np.zeros(len(check) + 1)
                r[0] = dirInd
                r[1:] = check
                subjectpics.append(r)
        return subjectpics



class noisedataset():

    def __init__(self,root_dir,trainsplit=.3, testsplit=.7, h = 6, w = 5, m = 0, s = 1, p = 4):
        self.train = []
        self.test = []
        s = 0
        self.h = h
        self.w = w
        self.ft = h * w
        self.root_dir = root_dir
        self.trainsplit = trainsplit
        self.testsplit = testsplit
        self.m = m
        self.s = s
        dirs = os.listdir(root_dir)
        dirs.remove(".DS_Store")
        self.dirs = dirs
        with mp.Pool(mp.cpu_count()) as pool:
            for pics in tqdm(pool.imap(self.loadImages, range(len(dirs))), total=len(self.dirs)):
                s += 1
                ttrain,ttest = tts(pics,test_size=testsplit,train_size=trainsplit)

                self.train.append(ttrain)
                self.test.append(ttest)



        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.nsamps = self.train.shape[1]
        self.train = self.train.reshape((-1, h * w + 1))
        self.test = self.test.reshape((-1,h * w + 1))
        np.random.shuffle(self.test)
        #Gauss
        self.test += np.random.normal(self.m,self.s,self.test.shape)
        #Salt and pepper
        #for i in range(len(self.test)):
        #    ttest = self.test[i]
        #    rands = np.random.choice(len(ttest), size = len(ttest)//4).astype(int)
        #    for i in rands:
        #        ttest[int(i)] = 1
        #    rands = np.random.choice(len(ttest), size = len(ttest)//4).astype(int)
        #    for i in rands:
        #        ttest[int(i)] = 0
        #    ttest = ttest/np.linalg.norm(ttest)
        #    ttest /= np.amax(ttest)
        #    self.test[i] = ttest

        #    self.test[i] = ttest
    def loadImages(self, dirInd):
        dir = self.dirs[dirInd]
        subjectpics = []
        for f in os.listdir('ExtendedYaleB/'+dir+'/'):
            if(f[-3:] == 'pgm' and f[-5] != 't'):
                check = np.array(Image.open('ExtendedYaleB/' + dir + '/' + f).resize((self.h,self.w))).flatten().astype(float)
                check /= np.linalg.norm(check)
                r = np.zeros(len(check) + 1)
                r[0] = dirInd
                r[1:] = check
                subjectpics.append(r)
        return subjectpics





class celebdataset():

    def __init__(self,h = 6, w = 5):
        dirs = os.listdir('5-celebrity-faces-dataset/data/train')
        dirs.remove(".DS_Store")
        self.trainsizes = []
        i = 0
        self.train = []
        for celeb in dirs:
            trainPics = os.listdir('5-celebrity-faces-dataset/data/train/' + celeb)
            self.trainsizes.append(len(trainPics))
            for pic in trainPics:
                check = np.array(Image.open('5-celebrity-faces-dataset/data/train/' + celeb + '/' + pic).convert('L').resize((h,w))).flatten().astype(float)
                check /= np.linalg.norm(check)
                r = np.zeros(len(check) + 1)
                r[0] = i
                r[1:] = check
                self.train.append(r)
        self.train = np.array(self.train)


        dirs = os.listdir('5-celebrity-faces-dataset/data/val')
        self.test = []
        for celeb in dirs:
            testPics = os.listdir('5-celebrity-faces-dataset/data/val/' + celeb)
            for pic in testPics:
                check = np.array(Image.open('5-celebrity-faces-dataset/data/val/' + celeb + '/' + pic).convert('L').resize((h,w))).flatten().astype(float)

                r = np.zeros(len(check) + 1)
                r[0] = i
                r[1:] = check
                self.test.append(r)
        self.test = np.array(self.test)




