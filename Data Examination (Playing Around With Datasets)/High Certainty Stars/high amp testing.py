import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

makePlot = True

files = 'resultsAmpPeriod.csv'

os.makedirs(('High Amp Stars Testing/'), exist_ok=True)

try:
    highAmpData = pd.read_csv(files,skiprows=11,sep=',')
    print("11")
except Exception as e:
    try:
        highAmpData = pd.read_csv(files,skiprows=12,sep=',')
        print("12")
    except Exception as e:
        try:
            highAmpData = pd.read_csv(files,skiprows=13,sep=',')
            print("13")
        except Exception as e:
            print("None of those!")
            exit()
            raise
        raise
    raise

print(highAmpData.describe())
#sns.jointplot(x=highAmpData["ksAmpl"], y=highAmpData["aVar"], kind='kde', color="grey", space=0)
#plt.show()
idRemovedData = highAmpData.drop('# vivaID', axis = 1)

def corr(x, y, **kwargs):

    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))

    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)

# Create a pair grid instance
#data= df[df['year'] == 2007]
grid = sns.PairGrid(data= idRemovedData,
                    vars = ['ksAmpl', 'aVar', 'freqSTR', 'freqPDM', 'freqPLfi2','freqPKfi2', 'freqLSG'], size = 4)

# Map the plots to the locations
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_upper(corr)
grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'darkred')
plt.savefig("test correlations.png")
plt.show()
