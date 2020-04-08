import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import argparse
import hdbscan
import sklearn.cluster as cluster
import time
import numpy
from sklearn.impute import SimpleImputer as Imputer # for new versions for sklearn
import shap

#sns.set_context('poster')
#sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 40, 'linewidths':0}


magAplPlot = False

ap = argparse.ArgumentParser()
ap.add_argument("-O", "--OSARGCheck", type=bool, default=False,
    help="OSARG plotting y?n?.")
ap.add_argument("-m", "--mode", type=str, default='default',
    help="Mode to plot.")
args = vars(ap.parse_args())
OSARGCheck = args['OSARGCheck']
mode=args['mode']

#from pylab import rcParams
#rcParams['figure.figsize'] = 14, 10

os.makedirs(('Clustering Stars/'), exist_ok=True)

import random
def plot_clusters(data, columns , algorithm, args, kwds):
    #columns=data.columns
    #data = data.to_numpy()
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    print('Clustering took {:.2f} s'.format(end_time - start_time))
    for indexes1 in range(0, len(columns)):
        for indexes2 in range(0, len(columns)):
            plt.scatter(data.T[indexes1], data.T[indexes2], c=colors, **plot_kwds)
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(True)
            frame.axes.get_yaxis().set_visible(True)
            plt.xlabel(columns[indexes1])
            plt.ylabel(columns[indexes2])
            if columns[indexes2] == 'Avar':
                plt.yscale('log')
            if columns[indexes1] == 'period':
                plt.xscale('log')
            plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
            plt.savefig('Clustering Stars/'+columns[indexes1]+' VS '+columns[indexes2]+'.png')
            #plt.show()
            plt.clf()

    # cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    #
    # remadeData = pd.DataFrame(data, columns=columns)
    # colors = [cmaps[x] if x >= 0 else 'black' for x in labels]
    # print(data[0])
    # remadeData['colour'] = colors
    # g = sns.PairGrid(remadeData, hue='colour')
    # g = g.map(sns.scatterplot, linewidths=1, edgecolor="w", s=40)
    # g = g.add_legend()
    # plt.show()

#########################
# With colour data #
#########################

data = pd.read_csv('colourCarlosObjects.csv',skiprows=9,na_values=[-999,-9.99999500e+08],skipinitialspace=True)
data['zAperMag3'] = data['# zAperMag3']
print(data.columns)

data['ones'] = [1.]*len(data['Avar'])

data.loc[data['Ngoodmeasures'] < 40,'MainVarType'] = 'Spurious'
#unhelpfulCategories = ['Others','iC','V*','Planet','Microlens','ISM','Spurious','EB','X','E']
#data = data[~data['MainVarType'].isin(unhelpfulCategories)]
print(data['MainVarType'].value_counts())

data['ratio'] = data['HeightKfi2']/data['FAPcorrelation2']
data['PDMratio'] = data['HeightPDM']/data['FAPcorrelation2']
data = data[data['FlagFbias6'] < 5]
data = data[data['ratio'] > 0.8]
data = data[data['HeightLfl2'] > 0]
data = data[data['hmksPnt'] > 0]
data['period'] = np.ones(len(data['FreqLSG']))
data['period'] = data['period'].div(data['FreqLSG'])
data['logPeriod'] = np.log10(data['period'])
data['logAmp'] = np.log10(data['Avar'])

# data.drop(['MainVarType','sourceID', 'ra_J2000_', 'dec_J2000_','Ncorrelation2','Ngoodmeasures','# zAperMag3','FlagFbias6',
#        'FlagFbias7', 'FlagNfreq','FAPcorrelation2','ratio'],axis=1,inplace=True)

# data = data[['jmhPnt', 'hmksPnt',
#        'period',
#        'Avar'
#        ]]

data = data[['jmhPnt', 'hmksPnt',
       'logPeriod',
       'Avar','HeightLfl2','PDMratio'
       ]]

sample = data.sample(frac=1.)
columns = sample.columns

imp = Imputer(missing_values=numpy.nan, strategy='median')
sample = imp.fit_transform(sample)
plot_clusters(sample,columns, hdbscan.HDBSCAN, (), {'min_cluster_size':30, 'min_samples':2})
quit()


iris = sns.load_dataset("iris")
#print(iris)

vivaData = pd.read_csv('QualityGoodObjs.csv', skiprows=20)
crossmatchedData = pd.read_csv('cflvsc_r01_crosssources.dat',sep=',')
print(crossmatchedData.columns)
# crossmatchedData.drop(['#cflvsc.sourceID',' cflvsc.FlagDataType'],axis=1,inplace = True)
# crossmatchedData.drop([' cflvsc.OtherNames',' cflvsc.Period',' cflvsc.MainVarType',' cflvsc.OtherVarType',' cflvsc.ra(J2000)', ' cflvsc.dec(J2000)',
#        ' cflvsc.gl(J2000)', ' cflvsc.gb(J2000)', ' cflvsc.zAperMag3',
#        ' cflvsc.zAperMag3Err', ' cflvsc.yAperMag3', ' cflvsc.yAperMag3Err',
#        ' cflvsc.jAperMag3', ' cflvsc.jAperMag3Err', ' cflvsc.hAperMag3',
#        ' cflvsc.hAperMag3Err'],axis=1,inplace = True)

crossmatchedData = crossmatchedData[[' cflvsc.Kfi2', ' cflvsc.L2',
       ' cflvsc.FreqKfi2', ' cflvsc.HeightKfi2',
       ' cflvsc.FreqLfl2', ' cflvsc.HeightLfl2',
       ' cflvsc.FreqLSG', ' cflvsc.HeightLSG',
       ' cflvsc.FreqPDM', ' cflvsc.HeightPDM',
       ' cflvsc.FreqSTR', ' cflvsc.HeightSTR',
       ' cflvsc.Avar']]

sample = crossmatchedData.sample(frac=0.1)

columns = sample.columns

from sklearn.impute import SimpleImputer as Imputer # for new versions for sklearn
imp = Imputer(missing_values=-999, strategy='median')
sample = imp.fit_transform(sample)

plot_clusters(sample,columns, hdbscan.HDBSCAN, (), {'min_cluster_size':50})
quit()

os.makedirs(('Selected Verified Crossmatched Stars/'), exist_ok=True)
os.makedirs(('Selected Verified Crossmatched Stars/PERIOD CUTS/'), exist_ok=True)

crossmatchedData = pd.read_csv('cflvsc_r01_crosssources.dat',sep=',')

#[LPV, M, RGB, SR, PUL, L, PER]
lpvPossibleObjects = ['  LPV','  M' ,'  RGB','  SR','  L']
lpvCandidates = crossmatchedData[crossmatchedData[' cflvsc.MainVarType'].isin(lpvPossibleObjects)]
print(lpvCandidates)
lpvCandidates = crossmatchedData[crossmatchedData[' cflvsc.OtherVarType'].str.contains('  LPV|  M|  RGB|  SR|  PUL|  L|  PER|  VAR')]
print(lpvCandidates)
print(lpvCandidates.columns)

literaturePeriods = lpvCandidates[lpvCandidates[' cflvsc.Period'] > 0]

###################################
# Compilation of Trimming Factors #
###################################

def crossmatchedNames(frequency):
    frequencyName = frequency+' period'
    logName = frequency + ' log10_period'
    columnName = frequency.replace("Freq",'Height')
    thresholdName = columnName+'-FAP'
    logHeight = 'log'+columnName
    return frequencyName,logName,columnName,thresholdName,logHeight

def vivaNames(frequency):
    #This is for the complete viva data
    if (frequency == ' cflvsc.FreqKfi2'):
        vivaName = frequency.replace(" cflvsc.Freq",'freqP')
    elif frequency == ' cflvsc.FreqLfl2':
        vivaName = frequency.replace(" cflvsc.Freq",'freqP')
        vivaName = vivaName.replace("l",'i')
    else:
        vivaName = frequency.replace(" cflvsc.Freq",'freq')
    vivaPeriod = vivaName+'Period'
    vivaLogPeriod = vivaName+'LogPeriod'
    vivaHeight = vivaName.replace("freq",'height')
    vivaThresholdName = vivaHeight+'-FAP'
    vivaLogHeight = 'log'+vivaHeight
    return vivaName,vivaPeriod,vivaLogPeriod,vivaHeight,vivaThresholdName,vivaLogHeight


import matplotlib.cm as cm
vivaData = pd.read_csv('AllGoodObjs.csv', skiprows=20)
compilationData = literaturePeriods[literaturePeriods[' cflvsc.Ngoodmeasures'] > 40]
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName,logName,columnName,thresholdName,logHeight = crossmatchedNames(frequency)
    #frequencyName = frequency+' period'
    compilationData[frequencyName] = np.ones(len(compilationData[frequency]))
    compilationData[frequencyName] = compilationData[frequencyName].div(compilationData[frequency])
    #logName = frequency + ' log10_period'
    compilationData[logName] = np.log10(compilationData[frequencyName])
    compilationData['crossmatch_1og10_period'] = np.log10(compilationData[' cflvsc.Period'])
    #columnName = frequency.replace("Freq",'Height')
    #thresholdName = columnName+'-FAP'
    compilationData[thresholdName] = compilationData[columnName]/compilationData[' cflvsc.FAPcorrelation2']
    compilationData[logHeight] = np.log10(compilationData[columnName])


    # #This is for the complete viva data
    # if (frequency == ' cflvsc.FreqKfi2'):
    #     vivaName = frequency.replace(" cflvsc.Freq",'freqP')
    # elif frequency == ' cflvsc.FreqLfl2':
    #     vivaName = frequency.replace(" cflvsc.Freq",'freqP')
    #     vivaName = vivaName.replace("l",'i')
    # else:
    #     vivaName = frequency.replace(" cflvsc.Freq",'freq')
    # vivaPeriod = vivaName+'Period'
    vivaName,vivaPeriod,vivaLogPeriod,vivaHeight,vivaThresholdName,vivaLogHeight = vivaNames(frequency)
    vivaData[vivaPeriod] = np.ones(len(vivaData[vivaName]))
    vivaData[vivaPeriod] = vivaData[vivaPeriod].div(vivaData[vivaName])
    #vivaLogPeriod = vivaName+'LogPeriod'
    vivaData[vivaLogPeriod] = np.log10(vivaData[vivaPeriod])
    #vivaHeight = vivaName.replace("freq",'height')
    #vivaThresholdName = vivaHeight+'-FAP'
    vivaData[vivaThresholdName] = vivaData[vivaHeight]/vivaData['faPcorrelation2']
    vivaData[vivaLogHeight] = np.log10(vivaData[vivaHeight])

    if mode == 'default':
        """Plot the viva data"""
        gridsize=60
        temp_data = vivaData[vivaData[vivaThresholdName] < 5]
        temp_data = temp_data[temp_data[vivaThresholdName] >= 0]
        plt.hexbin(temp_data[vivaLogPeriod], temp_data[vivaThresholdName], gridsize=gridsize, cmap=cm.jet, bins='log')
        #plt.hexbin(temp_data[vivaLogPeriod], temp_data[vivaThresholdName], gridsize=gridsize, cmap=cm.jet, bins=None)
        xedges = [temp_data[vivaLogPeriod].min(), temp_data[vivaLogPeriod].max()]
        yedges = [temp_data[vivaThresholdName].min(), temp_data[vivaThresholdName].max()]
        if yedges[-1] > 2:
            extent = [xedges[0], xedges[-1], yedges[0], 2]
            print(extent)
        else:
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            print(extent)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.axis(extent)
        plt.title('Scatter plot of power spectrum heights and periods')
        plt.ylabel(vivaThresholdName)
        plt.xlabel(vivaLogPeriod + ' & crossmatched period')

        cb = plt.colorbar()
        cb.set_label('Counts')
        #plt.show()

        gridsize=60
        temp_data = compilationData[compilationData[thresholdName] < 5]
        temp_data = temp_data[temp_data[thresholdName] >= 0]

        red_high = ((0., 0., 0.),
             (.3, .5, 0.5),
             (1., 1., 1.))
        blue_middle = ((0., .2, .2),
                 (.3, .5, .5),
                 (.8, .2, .2),
                 (1., .1, .1))
        green_none = ((0,0,0),(1,0,0))

        cdict3 = {'white':  red_high,

             'green': green_none,

             'grey': blue_middle,

             'alpha': ((0.0, 0.0, 0.0),
                       (0.3, 0.5, 0.5),
                       (1.0, 1.0, 1.0))
            }

        dark_low = ((0., 1., 1.),
             (1., 0., 0.))

        # dark_low = ((1., 1., 1.),
        #      (.6, 1., 0.),
        #      (.3, 1., 0.),
        #      (0., 0., 0.))

        dark_high = ((0., 0., 0.),
             (0.5, 0.5, 0.5),
             (1., 1., 1.))

        green_none = ((0,0,0),(1,0,0))

        cdict = {'red':  dark_high,

             'green': green_none,

             'blue': dark_low,

             'alpha': ((0.0, 0.0, 0.0),
                       (0.3, 0.0, 1.0),
                       (1.0, 1.0, 1.0))
            }

        cdict3 = {'red':  dark_low,

             'green': dark_low,

             'blue': dark_low,

             'alpha': ((0.0, 0.0, 0.0),
                       (0.3, 0.0, 1.0),
                       (1.0, 1.0, 1.0))
            }

        from matplotlib.cm import Greys
        import matplotlib.colors as clr
        cmap = clr.LinearSegmentedColormap.from_list('custom blue',
                                                 [(0,    '#ffff00'),
                                                  (0.25, '#002266'),
                                                  (1,    '#002266')], N=256)
        import matplotlib.pylab as pl
        from matplotlib.colors import ListedColormap

        cmap = pl.cm.Greys

        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))
        #cmap.N = 256

        # Set alpha
        my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
        #zeroBaseline = 10
        #my_cmap[:,-1] = np.concatenate(([0.1]*zeroBaseline),np.ones(cmap.N-zeroBaseline)))

        # Create new colormap
        my_cmap = ListedColormap(my_cmap)

        greys = LinearSegmentedColormap('Greys', cdict)
        plt.register_cmap(cmap=greys)

        dropout_high = LinearSegmentedColormap('Dropout', cdict3)
        plt.register_cmap(cmap = dropout_high)

        xName = 'crossmatch_1og10_period'
        xName = logName
        plt.hexbin(temp_data[xName], temp_data[thresholdName], gridsize=gridsize, cmap=my_cmap, bins=None)
        xedges = [temp_data[xName].min(), temp_data[xName].max()]
        yedges = [temp_data[thresholdName].min(), temp_data[thresholdName].max()]
        if yedges[-1] > 2:
            extent = [xedges[0], xedges[-1], yedges[0], 2]
            print(extent)
        else:
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            print(extent)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.axis(extent)

        cb = plt.colorbar()
        cb.set_label('mean value')
        plt.savefig('Selected Verified Crossmatched Stars/PERIOD CUTS/'+frequencyName+'ORIGINAL.png')
        #plt.show()
        plt.clf()
