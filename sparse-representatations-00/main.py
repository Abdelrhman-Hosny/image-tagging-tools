from recognizer import recognizer
from Dataset import dataset, noisedataset, celebdataset
from  tqdm import tqdm
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image


dset = None
FacialRecognition = None

def SCI(x, k = 5):
    currMax = None
    for i in range(28):
        start = i * dset.nsamps
        check = np.zeros(dset.nsamps * 28)
        check[start:start + dset.nsamps] = np.ones(dset.nsamps)
        portion = np.sum(x * check)/np.sum(x)
        if(currMax == None or portion > currMax):
            currMax = portion
    return ((k * currMax) - 1)/(k - 1)
def fullTest(h,w):
    global dset
    global FacialRecognition
    global goodfound
    global badfound
    dset = dataset('ExtendedYaleB/', trainsplit=.06, testsplit=.94, h = h, w = w)
    FacialRecognition = recognizer(dset)
    testSamples = len(dset.test)
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(trainingLoop, range(testSamples)),total=testSamples):
            if(res != -1):
                results.append(res)
    results = np.array(results)
    correct = np.sum(results)
    print('Correct:',correct)
    print('Total:', len(results))
    print('Accuracy:', correct / len(results))
    return correct / len(results)


def fullTestNoise(m, s):
    global dset
    global FacialRecognition
    dset = noisedataset('ExtendedYaleB/', trainsplit=.06, testsplit=.94, m = m, s = s)
    FacialRecognition = recognizer(dset)
    testSamples = len(dset.test)
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(trainingLoop, range(testSamples)),total=testSamples):
            if(res != -1):
                results.append(res)

    results = np.array(results)
    correct = np.sum(results)
    print('Correct:',correct)
    print('Total:', len(results))
    print('Accuracy:', correct / len(results))
    return correct / len(results)
def fullTestNoiseSP(p):
    global dset
    global FacialRecognition
    dset = noisedataset('ExtendedYaleB/', trainsplit=.06, testsplit=.94, p = p)
    FacialRecognition = recognizer(dset)
    testSamples = len(dset.test)
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(trainingLoop, range(testSamples)),total=testSamples):
            if(res != -1):
                results.append(res)

    results = np.array(results)
    correct = np.sum(results)
    print('Correct:',correct)
    print('Total:', len(results))
    print('Accuracy:', correct / len(results))
    return correct / len(results)
def trainingLoop(s=0):

    global dset
    global FacialRecognition
    eps = .05
    tau = .3
    x = np.random.default_rng().standard_normal(dset.nsamps * 28)
    sample = dset.test[s]
    img = sample[1:]
    label = int(sample[0])
    FacialRecognition.solve(x, img, eps)

    xhat = FacialRecognition.getOptim()
    if(type(xhat) == type(None)):
        return 0

    #FOR TESTING SCI

    #if(SCI(xhat) <= tau):
    #    return -1


    bestI = None
    bestRes = None
    for i in range(28):
        start = i * dset.nsamps
        check = np.zeros(dset.nsamps * 28)
        check[start:start + dset.nsamps] = np.ones(dset.nsamps)
        currRes = np.linalg.norm(img - (np.matmul(FacialRecognition.train, xhat * check)))
        if(bestI == None or currRes < bestRes):
            bestI = i
            bestRes = currRes
    if(bestI == label):
        return 1
    else:
        return 0



def all():
    tests = [(6,5), (9,7), (12,10), (15,12)]#, (20,17), (32,28)]
    #tests = [(6,5),(9,7)]

    accuracies = []
    for i in tests:
        print("Testing h =", i[0], "   w =", i[1])
        accuracies.append(fullTest(i[0], i[1]))

    print(accuracies)
    plt.figure()
    #plt.plot([30,63], accuracies)
    plt.plot([30,63,120,180], accuracies)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Feature Dimension')
    plt.show()

def allNoise():
    tests = [(0,1), (0,5), (0,10), (0,30), (0,60)]
    #tests = [40,30,20,10,15]
    accuracies = []
    for i in tests:
        print("Testing m =", i[0], "   s =", i[1])
        #print("Testing p =", i)
        accuracies.append(fullTestNoise(i[0], i[1]))
        #accuracies.append(fullTestNoiseSP(i))

    print(accuracies)
    plt.figure()
    plt.plot([1,5,10,30,60], accuracies)
    #plt.plot(tests, accuracies)
    plt.xlabel('Noise Portion')
    plt.ylabel('Accuracy')
    #plt.title('Accuracy vs Noise Portion')
    plt.title('Accuracy vs Standard Deviation')
    plt.show()

def reconstruction():
    global dset
    global FacialRecognition
    dset = dataset('ExtendedYaleB/', trainsplit=.06, testsplit=.94, h = 24, w=21)
    FacialRecognition = recognizer(dset)
    xhat = None
    while(type(xhat) == type(None)):

        choice = np.random.choice(len(dset.test))
        sample = dset.test[choice][1:]
        x = np.random.default_rng().standard_normal(dset.nsamps * 28)
        FacialRecognition.solve(x, sample, .05)
        xhat = FacialRecognition.getOptim()
        if(type(xhat) == type(None)):
            print("No solution")
            continue
        if(SCI(xhat) < .5):
            print("Solution found but not sparse")
            print(SCI(xhat))
            xhat = None
            continue
    recon = np.matmul(FacialRecognition.train, xhat)

    xhat /= np.amax(xhat)
    recon /= np.amax(recon)
    plt.figure()
    plt.subplot(121)
    plt.imshow(sample.reshape((24,21)), cmap='gray')
    plt.subplot(122)
    plt.imshow(recon.reshape((24,21)), cmap='gray')
    plt.show()


def celebTest(h, w):

    dset = celebdataset(h,w)
    FR = recognizer(dset)
    x = np.random.default_rng().standard_normal(len(dset.train))
    correct = 0
    for y in dset.test:
        label = int(y[0])
        img = y[1:]
        npics = dset.trainsizes[label]
        FR.solve(x,img,.05)
        xhat = FR.getOptim()
        if(type(xhat) == type(None)):
            continue
        currBest = None
        bestInd = None
        for i in range(5):
            check = np.zeros(len(dset.train))
            start = sum(dset.trainsizes[:i])
            end = start + npics
            check[start:end] = np.ones(end - start)
            currRes = np.linalg.norm(img - np.matmul(FR.train, xhat * check))
            if(currBest == None or currRes < currBest):
                currBest = currRes
                bestInd = i
        if(label == bestInd):
            correct += 1
    print("Correct:", correct)
    print("Total:", len(dset.test))
    print("Accuracy:", correct / len(dset.test))
#reconstruction()
#fullTest(12,10)
all()

#allNoise()
#for i in [(6, 5), (12,10), (24, 20)]:
#    celebTest(i[0], i[1])

#dset = celebdataset(100,80)
#img = dset.train[0][1:]

#plt.figure()
#plt.imshow(img.reshape((100,80)), cmap = 'gray')
#plt.show()

#dset = dataset('ExtendedYaleB/', trainsplit=.06, testsplit=.94, h = 6, w = 5)
#FR = recognizer(dset)
#x = np.random.default_rng().standard_normal(dset.nsamps * 28)
#bad = None
#badfound = False
#good = None
#goodfound = False
#for i in dset.test:
#    label = i[0]
#    img = i[1:]
#    FR.solve(x,img,.05)
#    xhat = FR.getOptim()
#    if(type(xhat) == type(None)):
#        continue
#    elif(SCI(xhat) < .3):
#        badfound = True
#        bad = xhat
#    elif(SCI(xhat) > .9):
#
#        goodfound = True
#        good = xhat
#    if(goodfound and badfound):
#        break
#
#plt.figure()
#plt.subplot(121)
#plt.plot(bad)
#plt.title('Bad Representation')
#plt.ylabel('Magnitude')
#plt.xlabel('Representation')
#plt.subplot(122)
#plt.plot(good)
#plt.xlabel('Representation')
#plt.ylabel('Magnitude')
#plt.title('Good Representation')
#plt.show()



#im = np.array(Image.open('ExtendedYaleB/yaleB11/yaleB11_P00A-005E-10.pgm'))
##im2 = im + np.random.normal(0,30,im.shape)
#a = im.shape
#im2 = im.flatten()
#
#salts = np.random.choice(len(im2), size = len(im2)//40)
#peppers = np.random.choice(len(im2), size=len(im2)//40)
#im2[salts] = 255
#im2[peppers] = 0
#im2 = im2.reshape(a)
#plt.figure()
#plt.subplot(121)
#plt.imshow(im, cmap ='gray')
#plt.subplot(122)
#plt.imshow(im2, cmap = 'gray')
#plt.show()

