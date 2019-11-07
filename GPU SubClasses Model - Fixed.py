# coding: utf-8
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Concatenate, Conv1D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.utils import multi_gpu_model

import tensorflow as tf #Getting tf not defined errors

from sklearn.model_selection import StratifiedKFold, train_test_split
#from tqdm import tqdm

import numpy as np
import pandas as pd
import glob, os, random
import argparse

from collections import Counter #Count number of objects in each class

from os.path import dirname as up

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,
    help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
num_gpu = args["gpus"]

## Settings
# Files Setting
limit = 7000 # Maximum amount of Star Per Class Per Survey
extraRandom = True
permutation = True # Permute Files
BALANCE_DB = True # Balance or not
maximum = 5

# Mini Settings
MAX_NUMBER_OF_POINTS = 500
NUMBER_OF_POINTS = 500
n_splits = 10 #Changing this for speed, just to test whether fixing the matrix shit works
#n_splits = 10
validation_set = 0.2

# Iterations
step = 250

# Network Settings
verbose = True
batch_size = 2056
dropout = 0.5
hidden_dims = 128
epochs = 100

# Convolutions
filters = 128
filters2 = 64
kernel_size = 50
kernel_size2 = 50

# Paths
NumberOfFiles = 'Subclasses'
base_path = os.getcwd() #Data is now stored one level up

from pathlib import Path
p = Path(base_path)
one_up = str(p.parent)
print("Moving one level up to: ", str(one_up))

#Laptop version
regular_exp1 = one_up + '/Data/OGLE/**/**/*.dat'
regular_exp2 = one_up + '/Data/VVV/**/**/**/*.csv'
regular_exp3 = one_up + '/Data/ASASSN-notfound/**/**/*.dat'
#regular_exp3 = one_up + '/Data/ASASSN/**/**/*.csv'

## Open Databases
#subclasses = ['cep10', 'cepF', 'RRab', 'RRc', 'nonEC', 'EC', 'Mira', 'SRV', 'Osarg']
subclasses = ['lpv','cep','rrlyr','ecl-c','ecl-nc']
#subclasses = ['clasCep','clasOvertoneCep','t2Cep','t2RVTau','contactBinary','detachedBinary','dsct','Mira','OSARG','SRV','RRab','RRc','RRd']
subclasses = ['clasCep','clasOvertoneCep','t2Cep','t2RVTau','contactBinary','detachedBinary','dsct','Mira','OSARG','SRV','RRab','RRc'] #Trying without RRd and only two surveys

#Make some fake classes and new fake data folders with just 0s and stuff to check it works
#subclasses = ['noise']
#regular_exp1

def get_filename(directory, N, early, activation='relu'):
    if activation == 'relu':
        directory += '/relu/'
    elif activation == 'sigmoid':
        directory += '/sigmoid/'
    else:
        directory += '/tanh/'

    if not os.path.exists(directory):
        print('[+] Creating Directory \n\t ->', directory)
        os.mkdir(directory)

    name = '1) Red ' + str(N)
    directory += '/'
    return directory, name

def get_files(extraRandom = False, permutation=False):
    files1 = np.array(list(glob.iglob(regular_exp1, recursive=True)))
    files2 = np.array(list(glob.iglob(regular_exp2, recursive=True)))
    files3 = np.array(list(glob.iglob(regular_exp3, recursive=True)))
    #Glob searches for all files that fit the format given in regular_exp1
    #Then puts them in a list

    print('[!] Files in Memory')
    print(files1)
    print(files2)
    print(files3)

    # Permutations
    if permutation:
        files1 = files1[np.random.permutation(len(files1))]
        files2 = files2[np.random.permutation(len(files2))]
        files3 = files3[np.random.permutation(len(files3))]

        print('[!] Permutation applied')
        #Shuffles the arrays

    aux_dic = {}
    ogle = {}
    ATLAS = {}
    vvv = {}
    asassn = {}
    for subclass in subclasses:
        aux_dic[subclass] = []
        ogle[subclass] = 0
        vvv[subclass] = 0
        asassn[subclass] = 0


    new_files = []
    #for idx in tqdm(range(len(files2))): #tqdm is a progress bar
    for idx in range(len(files1)): #tqdm is a progress bar
        foundOgle = False
        foundVista = False
        foundAsassn = False

        for subclass in subclasses:

            # Ogle
            # Limit is max stars of one class taken from survey (default 8000)
            if not foundOgle and ogle[subclass] < limit and subclass in files1[idx]:
                new_files += [[files1[idx], 0]]
                ogle[subclass] += 1
                foundOgle = True

            # VVV
            # idx check since VVV has less data than Ogle
            if not foundVista and vvv[subclass] < limit and idx < len(files2) and subclass in files2[idx] and os.path.getsize(files2[idx]) > 1000:
               new_files += [[files2[idx], 0]]
               vvv[subclass] += 1
               foundVista = True

            # ASASSN
            # some of the classes lack all 8000 objects
            if not foundAsassn and asassn[subclass] < limit and idx < len(files3) and subclass in files3[idx]:
               new_files += [[files3[idx], 0]]
               asassn[subclass] += 1
               foundAsassn = True

    del files1, files2, files3

    print('[!] Loaded Files')

    return new_files


def replicate_by_survey(files, yTrain):

    surveys = ["OGLE", "VVV", "ASASSN"]

    new_files = []
    for s in surveys:
        mask = [ s in i for i in yTrain]
        auxYTrain = yTrain[mask]

        new_files += replicate(files[mask])

    return new_files


def replicate(files):
    aux_dic = {}
    for subclass in subclasses:
        aux_dic[subclass] = []

    for file, num in files:
        for subclass in subclasses:
            if subclass in file:
                aux_dic[subclass].append([file, num])
                break

    new_files = []
    for subclass in subclasses:
        array = aux_dic[subclass]
        length = len(array)
        if length == 0:
            continue

        new_files += array
        if length < limit and extraRandom:
                count = 1
                q = limit // length
                for i in range(1, min(q, maximum)):
                    for file, num in array:
                        new_files += [[file, count]]
                    count += 1
                r = limit - q*length
                if r > 1:
                    new_files += [[random.choice(array)[0], count] for i in range(r)]

    return new_files

def get_survey(path):
    if 'VVV' in path:
        return 'VVV'
    elif 'OGLE' in path:
        return 'OGLE'
    elif 'ASASSN' in path:
        return 'ASASSN'
    else:
        return 'err'

def get_name(path):
    for subclass in subclasses:
        if subclass in path:
            return subclass
    return 'err'

def get_name_with_survey(path):
    for subclass in subclasses:
        if subclass in path:
            survey = get_survey(path)
            return survey + '_' + subclass
    return 'err'

def open_vista(path, num):
    df = pd.read_csv(path, comment='#', sep=',', header = None)
    #print(df.iloc[0])
    df.columns = ['sourceID','mjd','mag','ppErrBits','Flag','empty']
    df = df[df.mjd > 0]
    df = df.sort_values(by=[df.columns[1]])

    # Something related to 3 standard deviations
    #df = df[np.abs(df.mjd-df.mjd.mean())<=(3*df.mjd.std())]

    time = np.array(df[df.columns[1]].values, dtype=float)
    magnitude = np.array(df[df.columns[2]].values, dtype=float)
    error = np.array(df[df.columns[3]].values, dtype=float)

    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]

    # Num
    step = random.randint(1, 2)
    count = random.randint(0, num)

    time = time[::step]
    magnitude = magnitude[::step]
    error = error[::step]

    time = time[count:]
    magnitude = magnitude[count:]
    error = error[count:]

    # Get Name of Class
    # folder_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    # path, folder_name = os.path.split(folder_path)

    return time.astype('float'), magnitude.astype('float'), error.astype('float')

def open_ogle(path, num, n, columns):
    df = pd.read_csv(path, comment='#', sep='\s+', header=None)
    df.columns = ['a','b','c']
    df = df[df.a > 0]
    df = df.sort_values(by=[df.columns[columns[0]]])

    # Erase duplicates if it exist
    df.drop_duplicates(subset='a', keep='first')

    # 3 Desviaciones Standard
    #df = df[np.abs(df.mjd-df.mjd.mean())<=(3*df.mjd.std())]

    time = np.array(df[df.columns[columns[0]]].values, dtype=float)
    magnitude = np.array(df[df.columns[columns[1]]].values, dtype=float)
    error = np.array(df[df.columns[columns[2]]].values, dtype=float)

    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]

    # Num
    step = random.randint(1, 2)
    count = random.randint(0, num)

    time = time[::step]
    magnitude = magnitude[::step]
    error = error[::step]

    time = time[count:]
    magnitude = magnitude[count:]
    error = error[count:]


    if len(time) > n:
        time = time[:n]
        magnitude = magnitude[:n]
        error = error[:n]

    # Get Name of Class
    # folder_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    # path, folder_name = os.path.split(folder_path)

    return time, magnitude, error

def open_asassn(path, num, n, columns):
    try:
        df = pd.read_csv(path, comment='#', sep='\s+', header=None)
    except Exception as e:
        print("Crashing Path",path)
        exit()
    try:
        df.columns = ['a','b','c','d','e','f']
    except Exception as e:
        return np.zeros(1), np.zeros(1), np.zeros(1)

    df = df[df.a > 0]
    df = df.sort_values(by=[df.columns[columns[0]]])

    # Erase duplicates if it exist
    df.drop_duplicates(subset='a', keep='first')

    # 3 Desviaciones Standard
    #df = df[np.abs(df.mjd-df.mjd.mean())<=(3*df.mjd.std())]

    time = np.array(df[df.columns[columns[0]]].values, dtype=float)
    try:
        magnitude = np.array(df[df.columns[columns[1]]].values, dtype=float)
    except Exception as e:
        df[df.columns[columns[1]]] = df[df.columns[columns[1]]].str.replace(r"[><]",'')
        magnitude = np.array(df[df.columns[columns[1]]].values, dtype=float)
    error = np.array(df[df.columns[columns[2]]].values, dtype=float)

    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]

    # Num
    step = random.randint(1, 2)
    count = random.randint(0, num)

    time = time[::step]
    magnitude = magnitude[::step]
    error = error[::step]

    time = time[count:]
    magnitude = magnitude[count:]
    error = error[count:]


    if len(time) > n:
        time = time[:n]
        magnitude = magnitude[:n]
        error = error[:n]

    # Get Name of Class
    # folder_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    # path, folder_name = os.path.split(folder_path)

    return time, magnitude, error

# Data has the form (Points,(Delta Time, Mag, Error)) 1D
def create_matrix(data, N):
    try:
        aux = np.append([0], np.diff(data).flatten()) #This does the normalization - subtracting each point from its predecessor
    except Exception as e:
        print('Crashed at diff!')
        print('Value in Data List: ', data)
        exit()

    # Padding with zero if aux is not long enough
    if max(N-len(aux),0) > 0:
        aux = np.append(aux, [0]*(N-len(aux)))

    return np.array(aux[:N], dtype='float').reshape(-1,1)

def dataset(files, N):
    input_1 = []
    input_2 = []
    yClassTrain = []
    survey = []
    #for file, num in tqdm(files):
    for file, num in files:
        num = int(num)
        t, m, e, c, s = None, None, None, get_name(file), get_survey(file)
        if c in subclasses:
            if 'VVV' in file:
                t, m, e = open_vista(file, num)
            elif 'OGLE' in file:
                t, m, e = open_ogle(file, num, N, [0,1,2])
            elif 'ASASSN' in file:
                t, m, e = open_asassn(file, num, N, [0,2,3])
            if c in subclasses:
                input_1.append(create_matrix(t, N))
                input_2.append(create_matrix(m, N))
                yClassTrain.append(c)
                survey.append(s)
            else:
                print('\t [!] E2 File not passed: ', file, '\n\t\t - Class: ',  c)
        else:
            print('\t [!] E1 File not passed: ', file, '\n\t\t - Class: ',  c)
    return np.array(input_1), np.array(input_2), np.array(yClassTrain), np.array(survey)


## Keras Model
def get_model(N, classes, activation='relu'):
    conv1 = Conv1D(filters, kernel_size, activation='relu')
    conv2 = Conv1D(filters2, kernel_size2, activation='relu')

    # For Time Tower
    input1 = Input((N, 1))
    out1 = conv1(input1)
    out1 = conv2(out1)

    # For Magnitude Tower
    input2 = Input((N, 1))
    out2 = conv1(input2)
    out2 = conv2(out2)

    out = Concatenate()([out1, out2])
    out = Flatten()(out)
    out = Dropout(dropout)(out)
    out = Dense(hidden_dims, activation=activation)(out)
    out = Dropout(dropout)(out)
    out = Dense(len(classes), activation='softmax')(out)

    model = Model([input1, input2], out)

    return model

def class_to_vector(Y, classes):
    new_y = []
    for y in Y:
        aux = []
        for val in classes:
            if val == y:
                aux.append(1)
            else:
                aux.append(0)
        new_y.append(aux)
    return np.array(new_y)

def serialize_model(name, model):
    # Serialize model to JSON
    model_json = model.to_json()
    with open(name + '.json', "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(name + ".h5")

def experiment(directory, files, Y, classes, N, n_splits):
    # Iterating
    activations = ['tanh']
    earlyStopping = [False]

    #Iterate over the activation functions, but only tanh is used where
    #Since it obtained the best results
    for early in earlyStopping:
        for activation in activations:
            # try:
            print('\t\t [+] Training',
                  '\n\t\t\t [!] Early Stopping', early,
                  '\n\t\t\t [!] Activation', activation)

            #Retreives directory name which includes the activation func,
            #Creates one chosen activation func if it doesn't exist
            #name var = "1) Red" + N
            #N is the number of points
            direc, name =  get_filename(directory, N,
                                        early, activation)
            filename_exp = direc + name
            yPred = np.array([])
            yReal = np.array([])
            sReal = np.array([])

            modelNum = 0
            skf = StratifiedKFold(n_splits=n_splits)

            print('files',files)
            print('Y',Y)
            print('Y counts', Counter(Y))

            for train_index, test_index in skf.split(files, Y):
                dTrain, dTest = files[train_index], files[test_index]
                yTrain = Y[train_index]

                ##############
                ### Get DB ###
                ##############

                # Replicate Files
                dTrain = replicate_by_survey(dTrain, yTrain)

                # Get Database
                dTrain_1, dTrain_2, yTrain, _ = dataset(dTrain, N)
                dTest_1, dTest_2, yTest, sTest  = dataset(dTest, N)

                yReal = np.append(yReal, yTest)
                sReal = np.append(sReal, sTest)
                yTrain = class_to_vector(yTrain, classes)
                yTest = class_to_vector(yTest, classes)

                ################
                ## Tensorboard #
                ################

                tensorboard = TensorBoard(log_dir= direc + 'logs',
                                          write_graph=True, write_images=True)

                ################
                ##    Model   ##
                ################

                callbacks = [tensorboard]
                if early:
                    earlyStopping = EarlyStopping(monitor='val_loss', patience=3,
                                                  verbose=0, mode='auto')
                    callbacks.append(earlyStopping)

                if num_gpu <= 1:
                    print("[!] Training with 1 GPU")
                    model = get_model(N, classes, activation)
                else:
                    print("[!] Training with", str(num_gpu), "GPUs")

                    # We'll store a copy of the model on *every* GPU and then combine
                    # the results from the gradient updates on the CPU
                    with tf.device("/cpu:0"):
                        model = get_model(N, classes, activation)

                    # Make the model parallel
                    model = multi_gpu_model(model, gpus=num_gpu)

                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit([dTrain_1, dTrain_2], yTrain,
                          batch_size=batch_size * num_gpu, epochs=epochs,
                          validation_split=validation_set, verbose=1,
                          callbacks=callbacks)

                yPred = np.append(yPred, np.argmax(model.predict([dTest_1, dTest_2]), axis=1))

                #################
                ##  Serialize  ##
                #################

                modelDirectory = direc + 'model/'
                if not os.path.exists(modelDirectory):
                    print('[+] Creating Directory \n\t ->', modelDirectory)
                    os.mkdir(modelDirectory)

                serialize_model(modelDirectory + str(modelNum), model)
                modelNum += 1

                del dTrain, dTest, yTrain, yTest, model
                # break

            yPred = np.array([classes[int(i)]  for i in yPred])

            # Save Matrix
            print('\n \t\t\t [+] Saving Results in', filename_exp)
            np.save(filename_exp, [yReal, yPred, sReal])
            print('*'*30)
            # except Exception as e:
            #     print('\t\t\t [!] Fatal Error:\n\t\t', str(e))

print('[+] Getting Filenames')
files = np.array(get_files(extraRandom, permutation))
YSubClass = []
for file, num in files:
    YSubClass.append(get_name_with_survey(file))
YSubClass = np.array(YSubClass)

NUMBER_OF_POINTS = 500
while NUMBER_OF_POINTS <= MAX_NUMBER_OF_POINTS:

    # Create Folder
    directory = './Results' + NumberOfFiles
    if not os.path.exists(directory):
        print('[+] Creating Directory \n\t ->', directory)
        os.mkdir(directory)

    experiment(directory, files, YSubClass, subclasses, NUMBER_OF_POINTS, n_splits)
    NUMBER_OF_POINTS += step
