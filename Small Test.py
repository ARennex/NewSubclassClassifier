import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import glob, os, random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def confusion_matrix_test():
    read_in_data = np.load('ResultsSubclasses/tanh/1) Red 500.npy')
    y_actu = pd.Series(read_in_data[0], name='Actual')
    y_pred = pd.Series(read_in_data[1], name='Predicted')
    s_actu = pd.Series(read_in_data[2], name='Survey')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    print(df_conf_norm)
    df_confusion = pd.crosstab([s_actu,y_actu], y_pred, rownames=['Survey','Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

    # from pandas_ml import ConfusionMatrix
    # cm = ConfusionMatrix(y_actu, y_pred)
    # data = cm.print_stats()
    # np.savetxt('ResultsSubclasses/Pandas ML File.txt', data)

"""
Calculate the mean+std of ogle and asassn rrlyr
"""

base_path = os.getcwd() #Data is now stored one level up
p = Path(base_path)
one_up = str(p.parent)
print("Moving one level up to: ", str(one_up))

regular_exp1 = one_up + '/Data/OGLE/**/**/*.dat'
regular_exp2 = one_up + '/Data/VVV/**/**/**/*.csv'
regular_exp3 = one_up + '/Data/ASASSN/**/**/*.dat'

files1 = np.array(list(glob.iglob(regular_exp1, recursive=True)))
files2 = np.array(list(glob.iglob(regular_exp2, recursive=True)))
files3 = np.array(list(glob.iglob(regular_exp3, recursive=True)))

files1 = list(set(files1))
files2 = list(set(files2))
files3 = list(set(files3))
#print(len(files1),len(files2),len(files3))
#print(len(),len(list(set(files2))),len(list(set(files3))))

subclasses = ['clasCep','clasOvertoneCep','t2Cep','t2RVTau','contactBinary','detachedBinary','Mira','OSARG','SRV','RRab','RRc'] #Trying without RRd and dsct (dsct are too short period to be easily found)

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
    df = df[np.abs(df.mjd-df.mjd.mean())<=(4*df.mjd.std())]

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
    df = df[np.abs(df.a-df.a.mean())<=(4*df.a.std())]

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
    df = df[np.abs(df.a-df.a.mean())<=(4*df.a.std())]

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

def dataset(files, N, mode):
    vvv_mean,vvv_std,vvv_diff_mean,vvv_diff_std = [],[],[],[]
    ogle_mean,ogle_std,ogle_diff_mean,ogle_diff_std = [],[],[],[]
    asassn_mean,asassn_std,asassn_diff_mean,asassn_diff_std = [],[],[],[]
    total_subclasses=[]
    for file in files:
        num = 0 #No repeated files
        t, m, e, c, s = None, None, None, get_name(file), get_survey(file)
        if c in subclasses:
            if 'VVV' in file:
                t, m, e = open_vista(file, num)
            elif 'OGLE' in file:
                t, m, e = open_ogle(file, num, N, [0,1,2])
            elif 'ASASSN' in file:
                t, m, e = open_asassn(file, num, N, [0,2,3])
            if c in subclasses:
                if mode == 'time':
                    x=t
                else:
                    x=m
                diff = create_matrix(x, N)
                if 'VVV' in file:
                    vvv_mean.append(np.mean(x))
                    vvv_std.append(np.std(x))
                    vvv_diff_mean.append(np.mean(diff))
                    vvv_diff_std.append(np.std(diff))
                elif 'OGLE' in file:
                    ogle_mean.append(np.mean(x))
                    ogle_std.append(np.std(x))
                    ogle_diff_mean.append(np.mean(diff))
                    ogle_diff_std.append(np.std(diff))
                elif 'ASASSN' in file:
                    asassn_mean.append(np.mean(x))
                    asassn_std.append(np.std(x))
                    asassn_diff_mean.append(np.mean(diff))
                    asassn_diff_std.append(np.std(diff))
                total_subclasses.append(c)
            else:
                print('\t [!] E2 File not passed: ', file, '\n\t\t - Class: ',  c)
        else:
            print('\t [!] E1 File not passed: ', file, '\n\t\t - Class: ',  c)
    np.savetxt(mode+"-subclasses.txt",total_subclasses, fmt='%s')

    np.savetxt(mode+"-vvvmean.txt",vvv_mean)
    np.savetxt(mode+"-vvvstd.txt",vvv_std)
    np.savetxt(mode+"-vvvdiffmean.txt",vvv_diff_mean)
    np.savetxt(mode+"-vvvdiffstd.txt",vvv_diff_std)

    np.savetxt(mode+"-oglemean.txt",ogle_mean)
    np.savetxt(mode+"-oglestd.txt",ogle_std)
    np.savetxt(mode+"-oglediffmean.txt",ogle_diff_mean)
    np.savetxt(mode+"-oglediffstd.txt",ogle_diff_std)

    np.savetxt(mode+"-asassnmean.txt",asassn_mean)
    np.savetxt(mode+"-asassnstd.txt",asassn_std)
    np.savetxt(mode+"-asassndiffmean.txt",asassn_diff_mean)
    np.savetxt(mode+"-asassndiffstd.txt",asassn_diff_std)

total_files = np.concatenate((files1,files2,files3),axis=None)
#dataset(total_files,500,'time')
#dataset(total_files,500,'mag')

print("total file length: ", len(total_files))

ogle_length = 247402
asassn_length = 256478
vvv_length = 232739

from collections import defaultdict

def plotting(mode):
    loaded_subclass = np.genfromtxt('mag-subclasses.txt',dtype='str')
    ogle_means = np.loadtxt('mag-ogle'+mode+'.txt')
    asassn_means = np.loadtxt('mag-asassn'+mode+'.txt')
    vvv_means = np.loadtxt('mag-vvv'+mode+'.txt')
    print("lengths",len(ogle_means),len(asassn_means),len(vvv_means))

    ogle_length = len(ogle_means)
    asassn_length = len(asassn_means)
    vvv_length = len(vvv_means)

    ogle_dict = defaultdict(list)
    asassn_dict = defaultdict(list)
    vvv_dict = defaultdict(list)
    for single_sub in subclasses:
        ogle_dict[single_sub] = []
        asassn_dict[single_sub] = []
        vvv_dict[single_sub] = []

    for file_number in range(0,ogle_length):
        ogle_dict[loaded_subclass[file_number]].append(ogle_means[file_number])

    for file_number in range(0,asassn_length):
        asassn_dict[loaded_subclass[247402+file_number]].append(asassn_means[file_number])

    for file_number in range(0,vvv_length):
        if not np.isnan(vvv_means[file_number]):
            vvv_dict[loaded_subclass[247402+232739+file_number]].append(vvv_means[file_number])

    def reject_outliers(data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    for single_sub in subclasses:
        vvv_clipped = reject_outliers(np.array(vvv_dict[single_sub]),5.)

        #mean = np.mean(ogle_dict[single_sub]+asassn_dict[single_sub]+vvv_dict[single_sub])
        #std = np.std(ogle_dict[single_sub]+asassn_dict[single_sub]+vvv_dict[single_sub])
        mean = np.mean(np.concatenate([ogle_dict[single_sub],asassn_dict[single_sub],vvv_clipped]))
        std = np.std(np.concatenate([ogle_dict[single_sub],asassn_dict[single_sub],vvv_clipped]))
        #print(mean,std)
        #print(np.mean(ogle_dict[single_sub]),np.mean(asassn_dict[single_sub]),np.mean(vvv_clipped))
        #print(np.mean(ogle_dict[single_sub]),np.mean(asassn_dict[single_sub]),np.mean(vvv_dict[single_sub]))
        plt.hist(ogle_dict[single_sub], alpha=0.5, range=[mean-5*std,mean+5*std], label='OGLE data')
        plt.hist(asassn_dict[single_sub], alpha=0.5, range=[mean-5*std,mean+5*std], label='ASASSN data')
        plt.hist(vvv_clipped, alpha=0.5, range=[mean-5*std,mean+5*std], label='VVV data')
        #plt.hist(vvv_dict[single_sub], alpha=0.5, range=[mean-10*std,mean+10*std])
        plt.legend()
        plt.savefig('3 survey '+mode+single_sub+'.png')
        plt.close()

    # ogle_diff_std = np.loadtxt('mag-oglediffmean.txt')
    # asassn_diff_std = np.loadtxt('mag-asassndiffmean.txt')
    # vvv_diff_std = np.loadtxt('mag-vvvdiffmean.txt')
    #
    # ogle_length = len(ogle_diff_std)
    # asassn_length = len(asassn_diff_std)
    # vvv_length = len(vvv_diff_std)
    #
    # ogle_dict = defaultdict(list)
    # asassn_dict = defaultdict(list)
    # vvv_dict = defaultdict(list)
    # for single_sub in subclasses:
    #     ogle_dict[single_sub] = []
    #     asassn_dict[single_sub] = []
    #     vvv_dict[single_sub] = []
    #
    # for file_number in range(0,ogle_length):
    #     ogle_dict[loaded_subclass[file_number]].append(ogle_diff_std[file_number])
    #
    # for file_number in range(0,asassn_length):
    #     asassn_dict[loaded_subclass[247402+file_number]].append(asassn_diff_std[file_number])
    #
    # for file_number in range(0,vvv_length):
    #     if not np.isnan(vvv_diff_std[file_number]):
    #         vvv_dict[loaded_subclass[247402+232739+file_number]].append(vvv_diff_std[file_number])
    #
    # for single_sub in subclasses:
    #     vvv_clipped = reject_outliers(np.array(vvv_dict[single_sub]),5.)
    #     mean = np.mean(np.concatenate([ogle_dict[single_sub],asassn_dict[single_sub],vvv_clipped]))
    #     std = np.std(np.concatenate([ogle_dict[single_sub],asassn_dict[single_sub],vvv_clipped]))
    #     plt.hist(ogle_dict[single_sub], alpha=0.5, range=[mean-5*std,mean+5*std])
    #     plt.hist(asassn_dict[single_sub], alpha=0.5, range=[mean-5*std,mean+5*std])
    #     plt.hist(vvv_clipped, alpha=0.5, range=[mean-5*std,mean+5*std])
    #     plt.savefig('3 diff survey,'+single_sub+'.png')
    #     plt.close()

plotting('mean')
plotting('std')
plotting('diffmean')
plotting('diffstd')
