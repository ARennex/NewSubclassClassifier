import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

magAplPlot = False
OSARGCheck = True

def corr(x, y, **kwargs):

    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))

    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)

files = ['resultsNewOGLE.csv']
variableColumnName = [' cflvsc.MainVarType']
os.makedirs(('Verified Crossmatched Stars/'), exist_ok=True)

crossmatchedData = pd.read_csv('cflvsc_r01_crosssources.dat',sep=',')

#[LPV, M, RGB, SR, PUL, L, PER]
lpvPossibleObjects = ['  LPV','  M' ,'  RGB','  SR','  PUL','  L','  PER']
lpvCandidates = crossmatchedData[crossmatchedData[' cflvsc.MainVarType'].isin(lpvPossibleObjects)]
print(lpvCandidates)
lpvCandidates = crossmatchedData[crossmatchedData[' cflvsc.OtherVarType'].str.contains('  LPV|  M|  RGB|  SR|  PUL|  L|  PER')]
print(lpvCandidates)
print(lpvCandidates.columns)

###################################################
# Period Matching Literaturee #
###################################################

def Period_Matching_Literature(input_data):

    for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
        frequencyName = frequency+' period'
        input_data[frequencyName] = np.ones(len(input_data[frequency]))
        input_data[frequencyName] = input_data[frequencyName].div(input_data[frequency])

    d = {}
    # iterating through the elements of list
    for i in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
        d[i] = 0

    for index, row in input_data.iterrows():
        for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
            frequencyName = frequency+' period'
            for frequencyMultiple in [0.25,0.33,0.5,1.,2.0,3.0,4.0]:
                if abs(row[frequencyName]-frequencyMultiple*row[' cflvsc.Period']) < row[' cflvsc.Period']/10:
                    d[frequency] += 1

            ##########################################
            ## Check multiples/fractions of periods ##
            ##########################################

    print("Out of ", len(input_data), ", the different methods found periods accurate to within 10% this number of times: ")
    print(d)
    print("*"*50)

literaturePeriods = lpvCandidates[lpvCandidates[' cflvsc.Period'] > 0]
Period_Matching_Literature(literaturePeriods)

literaturePeriods = crossmatchedData[crossmatchedData[' cflvsc.Period'] > 0]
Period_Matching_Literature(literaturePeriods)

lpvPossibleObjects = ['  LPV','  M' ,'  RGB','  SR','  PUL','  L','  PER']
lpvCandidates = crossmatchedData[crossmatchedData[' cflvsc.MainVarType'].isin(lpvPossibleObjects)]
literaturePeriods = lpvCandidates[lpvCandidates[' cflvsc.Period'] > 0]
Period_Matching_Literature(literaturePeriods)

for lpvPossibleObjects in ['  LPV','  M' ,'  RGB','  SR','  PUL','  L','  PER']:
    lpvCandidates = crossmatchedData[crossmatchedData[' cflvsc.MainVarType'] == lpvPossibleObjects]
    literaturePeriods = lpvCandidates[lpvCandidates[' cflvsc.Period'] > 0]
    print("Testing Objects of Type:",lpvPossibleObjects)
    Period_Matching_Literature(literaturePeriods)

#
# for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
#     frequencyName = frequency+' period'
#     crossmatchedData[frequencyName] = np.ones(len(crossmatchedData[frequency]))
#     crossmatchedData[frequencyName] = crossmatchedData[frequencyName].div(crossmatchedData[frequency])
#
# crossmatchedData = crossmatchedData[crossmatchedData[' cflvsc.Period'] > 0]
# crossmatchedData = crossmatchedData[crossmatchedData[' cflvsc.Period'] < 1000]
#
# # Create a pair grid instance
# #data= df[df['year'] == 2007]
# grid = sns.PairGrid(data= crossmatchedData,
#                     vars = [' cflvsc.Period', ' cflvsc.FreqSTR period', ' cflvsc.FreqPDM period', ' cflvsc.FreqLSG period', ' cflvsc.FreqKfi2 period',' cflvsc.FreqLfl2 period'], size = 4)
#
# # Map the plots to the locations
# grid = grid.map_upper(plt.scatter, color = 'darkred')
# grid = grid.map_upper(corr)
# grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
# grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'darkred')
# plt.savefig("test correlations.png")
# plt.show()

exit()

###################################################
# Ks Mag against Amplitude #
###################################################

if magAplPlot == True:

    os.makedirs(('Verified Crossmatched Stars/ksMag/'), exist_ok=True)
    lpvCandidates = lpvCandidates[lpvCandidates[' cflvsc.ksEMeanMagPawprint'] < 100]
    lpvCandidates = lpvCandidates[lpvCandidates[' cflvsc.ksEMeanMagPawprint'] > 0]
    print(lpvCandidates)

    # magnitudes = [' cflvsc.zAperMag3', ' cflvsc.yAperMag3',' cflvsc.jAperMag3',' cflvsc.hAperMag3',' cflvsc.ksAperMag3']
    # for mags in magnitudes:
    #     temp = lpvCandidates[lpvCandidates[mags] < 100]
    #     temp = temp[temp[mags] > 0]
    #     print(temp)

    temp = lpvCandidates[['#cflvsc.sourceID',' cflvsc.ksAperMag3',' cflvsc.ksEMeanMagPawprint',' cflvsc.MainVarType']]
    temp.to_csv("missing lpvs.csv",index=False)

    #plt.scatter(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], s=70, alpha=0.03)
    sns.lmplot( x=" cflvsc.Avar", y=" cflvsc.ksEMeanMagPawprint", data=lpvCandidates, fit_reg=False, hue=' cflvsc.MainVarType', legend=False)

    # Move the legend to an empty part of the plot
    plt.legend(loc='lower right')
    plt.savefig("Verified Crossmatched Stars/ksMag/ksmag VS amplitude.png")
    plt.clf()
    #plt.show()

    colours = ['r','b','g']
    subclass = np.array(lpvCandidates[variableColumnName[0]].unique())
    colours = sns.color_palette("coolwarm", len(subclass))
    print(colours)
    for subcounter in range(0,len(subclass)):
        temp = lpvCandidates[variableColumnName[0]] == subclass[subcounter]
        temp_data = lpvCandidates[temp]
        grid = sns.jointplot(x=" cflvsc.Avar", y=" cflvsc.ksEMeanMagPawprint", data=temp_data, kind='kde', space=0, color = colours[subcounter])
        plt.savefig("Verified Crossmatched Stars/ksMag/"+subclass[subcounter]+"ksmag VS amplitude.png")
        plt.clf()

###################################################
# Log Period against Amplitude #
###################################################

os.makedirs(('Verified Crossmatched Stars/Avar/'), exist_ok=True)

#print("Filtering file: ", files[counter-1])
#print(len(crossmatchedData['kFi2']))
lpvCandidates = lpvCandidates[lpvCandidates[' cflvsc.FlagFbias7'] >= 1]
lpvCandidates = lpvCandidates[lpvCandidates[' cflvsc.FlagFbias7'] < 10]
#crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] < 20]
#crossmatchedData = crossmatchedData[crossmatchedData['wesenheit'] > 0]
#print(len(crossmatchedData['kFi2']))
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    logName = frequency + ' log10_period'
    lpvCandidates[frequencyName] = np.ones(len(lpvCandidates[frequency]))
    lpvCandidates[frequencyName] = lpvCandidates[frequencyName].div(lpvCandidates[frequency])
    lpvCandidates[logName] = np.log10(lpvCandidates[frequencyName])

    sns.lmplot( x=logName, y=" cflvsc.Avar", data=lpvCandidates, fit_reg=False, hue=" cflvsc.ksEMeanMagPawprint", legend=False)
    #sns.lmplot( x=logName, y=" cflvsc.Avar", data=lpvCandidates, fit_reg=False, hue=variableColumnName[0], legend=False)

    # Move the legend to an empty part of the plot
    plt.legend(loc='upper right')
    plt.savefig("Verified Crossmatched Stars/Avar/"+frequency+"period VS amplitude.png")
    plt.clf()
    #plt.show()


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
