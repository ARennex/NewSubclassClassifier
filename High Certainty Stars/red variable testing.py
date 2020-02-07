import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

makePlot = True
plotGraph = False
files = 'resultsLikely.csv'

highAmpData = pd.read_csv(files,skiprows=13,sep=',')

likelihood_clipped = highAmpData[highAmpData['kFi2']/highAmpData['faPcorrelation2'] > 1]
likelihood_clipped = likelihood_clipped[likelihood_clipped['wesenheit'] > 0]
likelihood_clipped = likelihood_clipped[likelihood_clipped['wesenheit'] < 10]
# likelihood_clipped = likelihood_clipped[likelihood_clipped['wesenheit'] > 7.5]
# likelihood_clipped = likelihood_clipped[likelihood_clipped['wesenheit'] < 15.5]

#######################################################
## There is a concerning negative value of this flag ##
## It occured at the location of an aliasing spike   ##
#######################################################
#print(likelihood_clipped[(likelihood_clipped['period'] >= 360) & (likelihood_clipped['period'] <= 370)]['flagFbias7'])
#dataArtifact = likelihood_clipped[ (likelihood_clipped['period'] >= 360) & (likelihood_clipped['period'] <= 370) ].index
#likelihood_clipped.drop(dataArtifact , inplace=True)
likelihood_clipped = likelihood_clipped[likelihood_clipped['flagFbias7'] >= 1]

print(highAmpData)

if plotGraph:
    for frequency in ['freqSTR','freqPDM','freqLSG']:
        likelihood_clipped['period'] = np.ones(len(likelihood_clipped[frequency]))
        likelihood_clipped['period'] = likelihood_clipped['period'].div(likelihood_clipped[frequency])
        per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
        # per_clipped = likelihood_clipped[likelihood_clipped['period'] > 5]
        per_clipped['log10_period'] = np.log10(per_clipped['period'])

        print(per_clipped)

        g = sns.jointplot(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], kind='hex')
        g.ax_joint.invert_yaxis()
        plt.show()

        plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
        plt.gca().invert_yaxis()
        plt.show()

        g = sns.jointplot(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], kind='kde', color="grey", space=0)
        g.ax_joint.invert_yaxis()
        plt.show()

if makePlot:
    frequency = 'freqPDM'
    likelihood_clipped['period'] = np.ones(len(likelihood_clipped[frequency]))
    likelihood_clipped['period'] = likelihood_clipped['period'].div(likelihood_clipped[frequency])
    per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
    per_clipped['log10_period'] = np.log10(per_clipped['period'])

    plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.1)
    plt.gca().invert_yaxis()
    plt.savefig('WesenheitVSPeriod Focused 3.png')
    plt.show()

    g = sns.jointplot(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], kind='kde', space=0)
    g.ax_joint.invert_yaxis()
    g.savefig('WesenheitVSPeriod Focused.png')
    plt.show()

    g = sns.jointplot(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], kind='hex')
    g.ax_joint.invert_yaxis()
    g.savefig('WesenheitVSPeriod Focused 2.png')
    plt.show()
