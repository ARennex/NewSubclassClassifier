import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

makePlot = True

def corr(x, y, **kwargs):

    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))

    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)

files = ['resultsAsassn.csv','resultsOGLE.csv','resultsOGLEExpanded.csv']

os.makedirs(('Crossmatched Stars Testing/'), exist_ok=True)

for counter in [1,2,3]:

    crossmatchedData = pd.read_csv(files[counter-1],skiprows=15,sep=',')

    print(crossmatchedData.describe())

    idRemovedData = crossmatchedData[['freqSTR','freqPDM','freqLSG','ksAmpl','jmhPnt','kFi2','faPcorrelation2','wesenheit','ksAperMag3','flagNfreq','flagFbias7']]

    # Create a pair grid instance
    #data= df[df['year'] == 2007]
    grid = sns.PairGrid(data= idRemovedData,
                        vars = ['freqSTR','freqPDM','freqLSG','ksAmpl','jmhPnt','kFi2','faPcorrelation2','wesenheit'], size = 4)

    # Map the plots to the locations
    grid = grid.map_upper(plt.scatter, color = 'darkred')
    grid = grid.map_upper(corr)
    grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
    grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'darkred')
    plt.savefig("test correlations"+str(counter)+".png")
    plt.show()
