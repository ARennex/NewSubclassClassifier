import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

makePlot = False
OSARGCheck = True

def corr(x, y, **kwargs):

    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))

    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)

files = ['resultsAsassn.csv','resultsOGLE.csv','resultsOGLEExpanded.csv']
files = ['resultsNewOGLE.csv']
variableColumnName = ['vartype','varType','varType']
variableColumnName = ['varType']
os.makedirs(('Crossmatched Stars Testing/'), exist_ok=True)

if OSARGCheck == True:
    crossmatchedData = pd.read_csv('resultsOGLEExpanded.csv',skiprows=15,sep=',')
    crossmatchedData = crossmatchedData[crossmatchedData['ksAmpl'] > 2]
    crossmatchedData = crossmatchedData[crossmatchedData['varType'] == 'OSARG']
    crossmatchedData.to_csv('Outlier OSARGs.csv')
    print(crossmatchedData)
    #exit()

###################################################
# Ks Mag against Amplitude #
###################################################

os.makedirs(('Crossmatched Stars Testing/ksMag/'), exist_ok=True)
crossmatchedData = pd.read_csv('resultsNewOGLE.csv',skiprows=16,sep=',')
print(crossmatchedData)
crossmatchedData = crossmatchedData[crossmatchedData['ksAperMag3'] < 100]
crossmatchedData = crossmatchedData[crossmatchedData['ksAperMag3'] > 0]
print(crossmatchedData)

#plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
sns.lmplot( x="aVar", y="ksAperMag3", data=crossmatchedData, fit_reg=False, hue=variableColumnName[0], legend=False)

# Move the legend to an empty part of the plot
plt.legend(loc='lower right')
plt.savefig("Crossmatched Stars Testing/ksMag/ksmag VS amplitude.png")
plt.clf()
#plt.show()

colours = ['r','b','g']
subclass = np.array(crossmatchedData[variableColumnName[0]].unique())
for subcounter in [1,2,3]:
    temp = crossmatchedData[variableColumnName[0]] == subclass[subcounter-1]
    temp_data = crossmatchedData[temp]
    grid = sns.jointplot(x="aVar", y="ksAperMag3", data=temp_data, kind='kde', space=0, color = colours[subcounter-1])
    plt.savefig("Crossmatched Stars Testing/ksMag/"+subclass[subcounter-1]+"ksmag VS amplitude.png")
    plt.clf()

os.makedirs(('Crossmatched Stars Testing/ksMag Mag Limited/'), exist_ok=True)
crossmatchedData = pd.read_csv('resultsNewOGLE.csv',skiprows=16,sep=',')
print(crossmatchedData)
crossmatchedData = crossmatchedData[crossmatchedData['ksAperMag3'] < 100]
crossmatchedData = crossmatchedData[crossmatchedData['ksAperMag3'] > 12]
print(crossmatchedData)

#plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
sns.lmplot( x="aVar", y="ksAperMag3", data=crossmatchedData, fit_reg=False, hue=variableColumnName[0], legend=False)

# Move the legend to an empty part of the plot
plt.legend(loc='lower right')
plt.savefig("Crossmatched Stars Testing/ksMag Mag Limited/ksmag VS amplitude.png")
plt.clf()
#plt.show()

colours = ['r','b','g']
subclass = np.array(crossmatchedData[variableColumnName[0]].unique())
for subcounter in [1,2,3]:
    temp = crossmatchedData[variableColumnName[0]] == subclass[subcounter-1]
    temp_data = crossmatchedData[temp]
    grid = sns.jointplot(x="aVar", y="ksAperMag3", data=temp_data, kind='kde', space=0, color = colours[subcounter-1])
    plt.savefig("Crossmatched Stars Testing/ksMag Mag Limited/"+subclass[subcounter-1]+"ksmag VS amplitude.png")
    plt.clf()

exit()

###################################################
# Log Period against Amplitude #
###################################################

os.makedirs(('Crossmatched Stars Testing/aVar/'), exist_ok=True)

crossmatchedData = pd.read_csv('resultsNewOGLE.csv',skiprows=16,sep=',')

#print("Filtering file: ", files[counter-1])
#print(len(crossmatchedData['kFi2']))
#crossmatchedData = crossmatchedData[crossmatchedData['flagFbias7'] >= 1]
#crossmatchedData = crossmatchedData[crossmatchedData['flagFbias7'] < 10]
#crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] < 20]
#crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] > 0]
#print(len(crossmatchedData['kFi2']))

for frequency in ['freqSTR','freqPDM','freqLSG','freqPKfi2','freqPLfi2']:
    frequencyName = frequency+' period'
    logName = frequency + ' log10_period'
    crossmatchedData[frequencyName] = np.ones(len(crossmatchedData[frequency]))
    crossmatchedData[frequencyName] = crossmatchedData[frequencyName].div(crossmatchedData[frequency])
    #per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
    crossmatchedData[logName] = np.log10(crossmatchedData[frequencyName])

    #plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
    sns.lmplot( x=logName, y="aVar", data=crossmatchedData, fit_reg=False, hue=variableColumnName[0], legend=False)

    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')
    plt.savefig("Crossmatched Stars Testing/aVar/"+frequency+"period VS amplitude.png")
    plt.clf()
    #plt.show()

    colours = ['r','b','g']
    subclass = np.array(crossmatchedData[variableColumnName[0]].unique())
    #print(subclass)
    for subcounter in [1,2,3]:
        #print(variableColumnName[counter-1])
        #print(crossmatchedData[variableColumnName[counter-1]])
        #print(subclass[subcounter-1])
        temp = crossmatchedData[variableColumnName[0]] == subclass[subcounter-1]
        print(temp)
        temp_data = crossmatchedData[temp]
        grid = sns.jointplot(x=logName, y="aVar", data=temp_data, kind='kde', space=0, color = colours[subcounter-1])
        #g = grid.plot_joint(sns.scatterplot, data=crossmatchedData)
        #sns.kdeplot(tips.loc[tips['smoker']=='Yes', 'total_bill'], ax=g.ax_marg_x, legend=False)
        #sns.kdeplot(tips.loc[tips['smoker']=='No', 'total_bill'], ax=g.ax_marg_x, legend=False)
        #sns.kdeplot(tips.loc[tips['smoker']=='Yes', 'tip'], ax=g.ax_marg_y, vertical=True, legend=False)
        #sns.kdeplot(tips.loc[tips['smoker']=='No', 'tip'], ax=g.ax_marg_y, vertical=True, legend=False)
        plt.savefig("Crossmatched Stars Testing/aVar/"+subclass[subcounter-1]+frequency+"period VS amplitude.png")
        plt.clf()

exit()

for counter in [1,2,3]:
    crossmatchedData = pd.read_csv(files[counter-1],skiprows=15,sep=',')

    crossmatchedData['checksum'] = crossmatchedData['kFi2']/crossmatchedData['faPcorrelation2']
    for frequency in ['freqSTR','freqPDM','freqLSG']:
        crossmatchedData[frequency+' period'] = np.ones(len(crossmatchedData[frequency]))
        crossmatchedData[frequency+' period'] = crossmatchedData[frequency+' period'].div(crossmatchedData[frequency])
        #per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
        crossmatchedData['log10_period'] = np.log10(crossmatchedData[frequency+' period'])

        #plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
        sns.lmplot( x=frequency+" period", y="checksum", data=crossmatchedData, fit_reg=False, hue=variableColumnName[counter-1], legend=False)

        # Move the legend to an empty part of the plot
        plt.legend(loc='lower right')
        plt.savefig("Crossmatched Stars Testing/"+frequency+"period VS checksum"+str(counter)+".png")
        plt.clf()
        #plt.show()

##############################################
# Remove Biased Periods and repeat the above #
##############################################

os.makedirs(('Crossmatched Stars Testing/Without Aliased Periods/'), exist_ok=True)
for counter in [1,2,3]:
    crossmatchedData = pd.read_csv(files[counter-1],skiprows=15,sep=',')

    print("Filtering file: ", files[counter-1])
    print(len(crossmatchedData['kFi2']))
    crossmatchedData = crossmatchedData[crossmatchedData['flagFbias6'] >= 1]
    crossmatchedData = crossmatchedData[crossmatchedData['flagFbias6'] < 10]
    print(len(crossmatchedData['kFi2']))

    crossmatchedData['checksum'] = crossmatchedData['kFi2']/crossmatchedData['faPcorrelation2']
    for frequency in ['freqSTR','freqPDM','freqLSG']:
        crossmatchedData[frequency+' period'] = np.ones(len(crossmatchedData[frequency]))
        crossmatchedData[frequency+' period'] = crossmatchedData[frequency+' period'].div(crossmatchedData[frequency])
        #per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
        crossmatchedData['log10_period'] = np.log10(crossmatchedData[frequency+' period'])

        #plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
        sns.lmplot( x=frequency+" period", y="checksum", data=crossmatchedData, fit_reg=False, hue=variableColumnName[counter-1], legend=False)

        # Move the legend to an empty part of the plot
        plt.legend(loc='lower right')
        plt.savefig("Crossmatched Stars Testing/Without Aliased Periods/Trimmed "+frequency+"period VS checksum"+str(counter)+".png")
        plt.clf()
        #plt.show()

###################################################
# Remove Biased Periods and test with log periods #
###################################################

os.makedirs(('Crossmatched Stars Testing/Log Periods/'), exist_ok=True)
for counter in [1,2,3]:
    crossmatchedData = pd.read_csv(files[counter-1],skiprows=15,sep=',')

    print("Filtering file: ", files[counter-1])
    print(len(crossmatchedData['kFi2']))
    crossmatchedData = crossmatchedData[crossmatchedData['flagFbias6'] >= 1]
    crossmatchedData = crossmatchedData[crossmatchedData['flagFbias6'] < 10]
    print(len(crossmatchedData['kFi2']))

    crossmatchedData['checksum'] = crossmatchedData['kFi2']/crossmatchedData['faPcorrelation2']
    for frequency in ['freqSTR','freqPDM','freqLSG']:
        frequencyName = frequency+' period'
        logName = frequency + ' log10_period'
        crossmatchedData[frequencyName] = np.ones(len(crossmatchedData[frequency]))
        crossmatchedData[frequencyName] = crossmatchedData[frequencyName].div(crossmatchedData[frequency])
        #per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
        crossmatchedData[logName] = np.log10(crossmatchedData[frequencyName])

        #plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
        sns.lmplot( x=logName, y="checksum", data=crossmatchedData, fit_reg=False, hue=variableColumnName[counter-1], legend=False)

        # Move the legend to an empty part of the plot
        plt.legend(loc='lower right')
        plt.savefig("Crossmatched Stars Testing/Log Periods/"+frequency+"period VS checksum"+str(counter)+".png")
        plt.clf()
        #plt.show()

###################################################
# Log Period against Wesenheit #
###################################################

os.makedirs(('Crossmatched Stars Testing/Wesenheit/'), exist_ok=True)
for counter in [1,2,3]:
    crossmatchedData = pd.read_csv(files[counter-1],skiprows=15,sep=',')

    print("Filtering file: ", files[counter-1])
    print(len(crossmatchedData['kFi2']))
    crossmatchedData = crossmatchedData[crossmatchedData['flagFbias7'] >= 1]
    crossmatchedData = crossmatchedData[crossmatchedData['flagFbias7'] < 10]
    crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] < 20]
    crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] > 0]
    print(len(crossmatchedData['kFi2']))

    for frequency in ['freqSTR','freqPDM','freqLSG']:
        frequencyName = frequency+' period'
        logName = frequency + ' log10_period'
        crossmatchedData[frequencyName] = np.ones(len(crossmatchedData[frequency]))
        crossmatchedData[frequencyName] = crossmatchedData[frequencyName].div(crossmatchedData[frequency])
        #per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
        crossmatchedData[logName] = np.log10(crossmatchedData[frequencyName])

        #plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
        sns.lmplot( x=logName, y="wesenheit", data=crossmatchedData, fit_reg=False, hue=variableColumnName[counter-1], legend=False)

        # Move the legend to an empty part of the plot
        plt.legend(loc='lower right')
        plt.savefig("Crossmatched Stars Testing/Wesenheit/"+frequency+"period VS wesenheit"+str(counter)+".png")
        plt.clf()
        #plt.show()

###################################################
# Log Period against Amplitude #
###################################################

os.makedirs(('Crossmatched Stars Testing/Amplitude/'), exist_ok=True)
for counter in [1,2,3]:
    crossmatchedData = pd.read_csv(files[counter-1],skiprows=15,sep=',')

    #print("Filtering file: ", files[counter-1])
    #print(len(crossmatchedData['kFi2']))
    #crossmatchedData = crossmatchedData[crossmatchedData['flagFbias7'] >= 1]
    #crossmatchedData = crossmatchedData[crossmatchedData['flagFbias7'] < 10]
    #crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] < 20]
    #crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] > 0]
    #print(len(crossmatchedData['kFi2']))

    for frequency in ['freqSTR','freqPDM','freqLSG']:
        frequencyName = frequency+' period'
        logName = frequency + ' log10_period'
        crossmatchedData[frequencyName] = np.ones(len(crossmatchedData[frequency]))
        crossmatchedData[frequencyName] = crossmatchedData[frequencyName].div(crossmatchedData[frequency])
        #per_clipped = likelihood_clipped[likelihood_clipped['period'] > 20]
        crossmatchedData[logName] = np.log10(crossmatchedData[frequencyName])

        #plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
        sns.lmplot( x=logName, y="ksAmpl", data=crossmatchedData, fit_reg=False, hue=variableColumnName[counter-1], legend=False)

        # Move the legend to an empty part of the plot
        plt.legend(loc='lower right')
        plt.savefig("Crossmatched Stars Testing/Amplitude/"+frequency+"period VS amplitude"+str(counter)+".png")
        plt.clf()
        #plt.show()

        colours = ['r','b','g']
        subclass = np.array(crossmatchedData[variableColumnName[counter-1]].unique())
        #print(subclass)
        for subcounter in [1,2,3]:
            #print(variableColumnName[counter-1])
            #print(crossmatchedData[variableColumnName[counter-1]])
            #print(subclass[subcounter-1])
            temp = crossmatchedData[variableColumnName[counter-1]] == subclass[subcounter-1]
            print(temp)
            temp_data = crossmatchedData[temp]
            grid = sns.jointplot(x=logName, y="ksAmpl", data=temp_data, kind='kde', space=0, color = colours[subcounter-1])
            #g = grid.plot_joint(sns.scatterplot, data=crossmatchedData)
            #sns.kdeplot(tips.loc[tips['smoker']=='Yes', 'total_bill'], ax=g.ax_marg_x, legend=False)
            #sns.kdeplot(tips.loc[tips['smoker']=='No', 'total_bill'], ax=g.ax_marg_x, legend=False)
            #sns.kdeplot(tips.loc[tips['smoker']=='Yes', 'tip'], ax=g.ax_marg_y, vertical=True, legend=False)
            #sns.kdeplot(tips.loc[tips['smoker']=='No', 'tip'], ax=g.ax_marg_y, vertical=True, legend=False)
            plt.savefig("Crossmatched Stars Testing/Amplitude/"+subclass[subcounter-1]+frequency+"period VS amplitude"+str(counter)+".png")
            plt.clf()

if makePlot == True:
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
