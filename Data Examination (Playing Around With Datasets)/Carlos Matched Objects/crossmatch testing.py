import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

magAplPlot = False
OSARGCheck = True

sns.set_style('darkgrid')
def jointplot_w_hue(data, x, y, filename, hue=None, colormap = None,
                    figsize = None, fig = None, scatter_kws=None):
    #defaults
    if colormap is None:
        colormap = sns.color_palette() #['blue','orange']
    if figsize is None:
        figsize = (10,10)
    if fig is None:
        fig  = plt.figure(figsize = figsize)
    if scatter_kws is None:
        scatter_kws = dict(alpha=0.4, lw=1)

    # derived variables
    if hue is None:
        return "use normal sns.jointplot"
    hue_groups = data[hue].unique()

    subdata = dict()
    colors = dict()

    active_colormap = colormap[0: len(hue_groups)]
    legend_mapping = []
    for hue_grp, color in zip(hue_groups, active_colormap):
        legend_entry = mpatches.Patch(color=color, label=hue_grp)
        legend_mapping.append(legend_entry)

        subdata[hue_grp] = data[data[hue]==hue_grp]
        colors[hue_grp] = color

    # canvas setup
    grid = gridspec.GridSpec(2, 2,
                           width_ratios=[4, 1],
                           height_ratios=[1, 4],
                           hspace = 0, wspace = 0
                           )
    ax_main    = plt.subplot(grid[1,0])
    ax_xhist   = plt.subplot(grid[0,0], sharex=ax_main)
    ax_yhist   = plt.subplot(grid[1,1])#, sharey=ax_main)

    ## plotting

    # histplot x-axis
    for hue_grp in hue_groups:
        sns.distplot(subdata[hue_grp][x], color = colors[hue_grp]
                     , ax = ax_xhist)

    # histplot y-axis
    for hue_grp in hue_groups:
        sns.distplot(subdata[hue_grp][y], color = colors[hue_grp]
                     , ax = ax_yhist, vertical=True)

    # main scatterplot
    # note: must be after the histplots else ax_yhist messes up
    for hue_grp in hue_groups:
        sns.regplot(data = subdata[hue_grp], fit_reg=False,
                    x = x, y = y, ax = ax_main, color = colors[hue_grp]
                    , scatter_kws=scatter_kws
                   )

    # despine
    for myax in [ax_yhist, ax_xhist]:
        sns.despine(ax = myax, bottom=False, top=True, left = False, right = True
                    , trim = False)
        plt.setp(myax.get_xticklabels(), visible=False)
        plt.setp(myax.get_yticklabels(), visible=False)


    # topright
    ax_legend   = plt.subplot(grid[0,1])#, sharey=ax_main)
    plt.setp(ax_legend.get_xticklabels(), visible=False)
    plt.setp(ax_legend.get_yticklabels(), visible=False)

    ax_legend.legend(handles=legend_mapping)
    os.makedirs(('Verified Crossmatched Stars/' + filename + '/'), exist_ok=True)
    saveName = "Verified Crossmatched Stars/"+ filename + '/' + x + y + ".png"
    #saveName = "Verified Crossmatched Stars/"+ x + y + filename + ".png"
    plt.savefig(saveName)
    #plt.savefig("Verified Crossmatched Stars/"+ x + y + "FrequencyHeight.png")
    plt.close()
    return dict(fig = fig, gridspec = grid)

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
lpvCandidates = crossmatchedData[crossmatchedData[' cflvsc.OtherVarType'].str.contains('  LPV|  M|  RGB|  SR|  PUL|  L|  PER|  VAR')]
print(lpvCandidates)
print(lpvCandidates.columns)

literaturePeriods = lpvCandidates[lpvCandidates[' cflvsc.Period'] > 0]

###################################
# Compilation of Trimming Factors #
###################################
compilationData = literaturePeriods[literaturePeriods[' cflvsc.Ngoodmeasures'] > 40]
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    compilationData[frequencyName] = np.ones(len(compilationData[frequency]))
    compilationData[frequencyName] = compilationData[frequencyName].div(compilationData[frequency])

    logName = frequency + ' log10_period'
    compilationData[logName] = np.log10(compilationData[frequencyName])
    compilationData['crossmatch_1og10_period'] = np.log10(compilationData[' cflvsc.Period'])

    columnName = frequency.replace("Freq",'Height')

    # literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
    #                               q=[0, .2, .4, .6, .8, 1],
    #                               labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(compilationData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)

    compilationData['mod heights'] = pd.qcut(compilationData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print('*'*50)
    print(frequency)
    print('*'*50)
    print(results_table)
    print(compilationData['mod heights'].unique())
    print(compilationData['mod heights'].value_counts())

    jointplot_w_hue(data=compilationData, x = 'crossmatch_1og10_period', y = logName, filename='Compilation', hue = 'mod heights')['fig']

    os.makedirs(('Verified Crossmatched Stars/Height Histogram/'), exist_ok=True)
    temp = compilationData[columnName]
    hist = temp.hist(bins = 40)
    plt.title(columnName+ 'Histogram')
    plt.xlabel(columnName)
    plt.savefig('Verified Crossmatched Stars/Height Histogram/'+frequencyName+'.png')
    plt.clf()
    #plt.show()


import operator

def row_checker(row, freqName, heightName, lowerLimit, upperLimit):
    if (row[heightName] > lowerLimit) and ((row[heightName] < upperLimit)):
        return True
        #return row[freqName], row[heightName]
    else:
        return False
        #return -9.99E8, -9.99E8

ids = [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']
for index,row in compilationData.iterrows():
    for singleID in ids:
        heightName = singleID.replace("Freq",'Height')
        validFreq = singleID.replace(" cflvsc.", "valid")
        #validHeight = heightName.replace(" cflvsc.", "valid")

        if singleID == ' cflvsc.FreqSTR':
            valid = row_checker(row,singleID,heightName,0.05,10.0)
        elif singleID == ' cflvsc.FreqPDM':
            valid = row_checker(row,singleID,heightName,0.0,0.5)
        elif singleID == ' cflvsc.FreqLSG':
            valid = row_checker(row,singleID,heightName,4.5,1000.0)
        elif singleID == ' cflvsc.FreqKfi2':
            valid = row_checker(row,singleID,heightName,0.75,10.0)
        elif singleID == ' cflvsc.FreqLfl2':
            valid = row_checker(row,singleID,heightName,0.75,10.0)

        #########################################
        # This is where the new column is added #
        #########################################
        compilationData.loc[index,validFreq] = valid

print(compilationData)



#######################################
# 2nd Compilation of Trimming Factors #
#######################################
compilationData = compilationData[compilationData[' cflvsc.Ngoodmeasures'] > 50]
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    validFreq = frequency.replace(" cflvsc.", "valid")
    temporaryData = compilationData[compilationData[validFreq] == True]

    logName = frequency + ' log10_period'
    columnName = frequency.replace("Freq",'Height')

    # literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
    #                               q=[0, .2, .4, .6, .8, 1],
    #                               labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(temporaryData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)

    temporaryData['mod heights'] = pd.qcut(temporaryData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print('*'*50)
    print(frequency)
    print('*'*50)
    print(results_table)
    print(temporaryData['mod heights'].unique())
    print(temporaryData['mod heights'].value_counts())

    jointplot_w_hue(data=temporaryData, x = 'crossmatch_1og10_period', y = logName, filename='Version1', hue = 'mod heights')['fig']

    os.makedirs(('Verified Crossmatched Stars/Height Histogram/'), exist_ok=True)
    temp = temporaryData[columnName]
    hist = temp.hist(bins = 40)
    plt.title(columnName+ 'Histogram')
    plt.xlabel(columnName)
    plt.savefig('Verified Crossmatched Stars/Height Histogram/'+frequencyName+'.png')
    plt.clf()




##################################################################################
""""
# Not Just LPV stars #
"""
##################################################################################

crossmatchedData = pd.read_csv('cflvsc_r01_crosssources.dat',sep=',')


literaturePeriods = crossmatchedData[crossmatchedData[' cflvsc.Period'] > 0]

###################################
# Compilation of Trimming Factors #
###################################
compilationData = literaturePeriods[literaturePeriods[' cflvsc.Ngoodmeasures'] > 40]
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    compilationData[frequencyName] = np.ones(len(compilationData[frequency]))
    compilationData[frequencyName] = compilationData[frequencyName].div(compilationData[frequency])

    logName = frequency + ' log10_period'
    compilationData[logName] = np.log10(compilationData[frequencyName])
    compilationData['crossmatch_1og10_period'] = np.log10(compilationData[' cflvsc.Period'])

    columnName = frequency.replace("Freq",'Height')

    # literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
    #                               q=[0, .2, .4, .6, .8, 1],
    #                               labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(compilationData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)

    compilationData['mod heights'] = pd.qcut(compilationData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print('*'*50)
    print(frequency)
    print('*'*50)
    print(results_table)
    print(compilationData['mod heights'].unique())
    print(compilationData['mod heights'].value_counts())

    jointplot_w_hue(data=compilationData, x = 'crossmatch_1og10_period', y = logName, filename='AllPeriodic', hue = 'mod heights')['fig']

    os.makedirs(('Verified Crossmatched Stars/All Periodic Height Histogram/'), exist_ok=True)
    temp = compilationData[columnName]
    hist = temp.hist(bins = 40)
    plt.title(columnName+ 'Histogram')
    plt.xlabel(columnName)
    plt.savefig('Verified Crossmatched Stars/All Periodic Height Histogram/'+frequencyName+'.png')
    plt.clf()
    #plt.show()

ids = [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']
for index,row in compilationData.iterrows():
    for singleID in ids:
        heightName = singleID.replace("Freq",'Height')
        validFreq = singleID.replace(" cflvsc.", "valid")
        #validHeight = heightName.replace(" cflvsc.", "valid")

        if singleID == ' cflvsc.FreqSTR':
            valid = row_checker(row,singleID,heightName,0.05,10.0)
        elif singleID == ' cflvsc.FreqPDM':
            valid = row_checker(row,singleID,heightName,0.0,0.5)
        elif singleID == ' cflvsc.FreqLSG':
            valid = row_checker(row,singleID,heightName,4.5,1000.0)
        elif singleID == ' cflvsc.FreqKfi2':
            valid = row_checker(row,singleID,heightName,0.75,10.0)
        elif singleID == ' cflvsc.FreqLfl2':
            valid = row_checker(row,singleID,heightName,0.75,10.0)

        #########################################
        # This is where the new column is added #
        #########################################
        compilationData.loc[index,validFreq] = valid

print(compilationData)

#######################################
# 2nd Compilation of Trimming Factors #
#######################################
compilationData = compilationData[compilationData[' cflvsc.Ngoodmeasures'] > 50]
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    validFreq = frequency.replace(" cflvsc.", "valid")
    temporaryData = compilationData[compilationData[validFreq] == True]

    logName = frequency + ' log10_period'
    columnName = frequency.replace("Freq",'Height')

    # literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
    #                               q=[0, .2, .4, .6, .8, 1],
    #                               labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(temporaryData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)

    temporaryData['mod heights'] = pd.qcut(temporaryData[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print('*'*50)
    print(frequency)
    print('*'*50)
    print(results_table)
    print(temporaryData['mod heights'].unique())
    print(temporaryData['mod heights'].value_counts())

    jointplot_w_hue(data=temporaryData, x = 'crossmatch_1og10_period', y = logName, filename='GoodPeriodicVersion1', hue = 'mod heights')['fig']

    os.makedirs(('Verified Crossmatched Stars/Good Periodic Height Histogram/'), exist_ok=True)
    temp = temporaryData[columnName]
    hist = temp.hist(bins = 40)
    plt.title(columnName+ 'Histogram')
    plt.xlabel(columnName)
    plt.savefig('Verified Crossmatched Stars/Good Periodic Height Histogram/'+frequencyName+'.png')
    plt.clf()



exit()
###################################
# This bit here needs repurposing -> above ^ #
# It's not useful to simply find the highest peak - they're not the same scale #
################################################################################

for index,row in compilationData.iterrows():
    stats = {}
    for singleID in ids:
        heightName = singleID.replace("Freq",'Height')
        stats[row[singleID]] = row[heightName]

    print(stats)
    print(max(stats.items(), key=operator.itemgetter(1))[0])

exit()

#####################################################################
# Check period vs literature period based on frequency spike height #
#####################################################################
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    literaturePeriods[frequencyName] = np.ones(len(literaturePeriods[frequency]))
    literaturePeriods[frequencyName] = literaturePeriods[frequencyName].div(literaturePeriods[frequency])

    logName = frequency + ' log10_period'
    literaturePeriods[logName] = np.log10(literaturePeriods[frequencyName])
    literaturePeriods['crossmatch_1og10_period'] = np.log10(literaturePeriods[' cflvsc.Period'])

    columnName = frequency.replace("Freq",'Height')

    # literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
    #                               q=[0, .2, .4, .6, .8, 1],
    #                               labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(lpvCandidates[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)

    literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print('*'*50)
    print(frequency)
    print('*'*50)
    print(results_table)
    print(literaturePeriods['mod heights'].unique())
    print(literaturePeriods['mod heights'].value_counts())

    jointplot_w_hue(data=literaturePeriods, x = 'crossmatch_1og10_period', y = logName, filename='FrequencyHeight', hue = 'mod heights')['fig']

##############################################################################
# Check period vs literature period based on frequency spike relative height #
##############################################################################
for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    literaturePeriods[frequencyName] = np.ones(len(literaturePeriods[frequency]))
    literaturePeriods[frequencyName] = literaturePeriods[frequencyName].div(literaturePeriods[frequency])

    logName = frequency + ' log10_period'
    literaturePeriods[logName] = np.log10(literaturePeriods[frequencyName])
    literaturePeriods['crossmatch_1og10_period'] = np.log10(literaturePeriods[' cflvsc.Period'])

    columnName = frequency.replace("Freq",'HeightKfi2to')

    # literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
    #                               q=[0, .2, .4, .6, .8, 1],
    #                               labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(lpvCandidates[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)

    literaturePeriods['mod heights'] = pd.qcut(lpvCandidates[columnName],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print('*'*50)
    print(frequency)
    print('*'*50)
    print(results_table)
    print(literaturePeriods['mod heights'].unique())
    print(literaturePeriods['mod heights'].value_counts())

    jointplot_w_hue(data=literaturePeriods, x = 'crossmatch_1og10_period', y = logName, filename='RelativeFrequencyHeight', hue = 'mod heights')['fig']


exit()

literaturePeriods['mod good obs'] = pd.qcut(lpvCandidates[' cflvsc.Ngoodmeasures'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=['20%', '40%', '60%', '80%', '100%'])
results, bin_edges = pd.qcut(lpvCandidates[' cflvsc.Ngoodmeasures'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=['20%', '40%', '60%', '80%', '100%'],
                              retbins = True)
results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                            columns=['Threshold', 'Tier'])
print(results_table)
print(literaturePeriods['mod good obs'].unique())
print(literaturePeriods['mod good obs'].value_counts())
for otherColumn in [' cflvsc.ksEMeanMagPawprint']:
    jointplot_w_hue(data=literaturePeriods, x = " cflvsc.Avar", y = otherColumn, hue = 'mod good obs')['fig']

lpvCandidates = lpvCandidates[lpvCandidates[' cflvsc.Ngoodmeasures'] > 50]

literaturePeriods['mod good obs'] = pd.qcut(lpvCandidates[' cflvsc.Ngoodmeasures'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=['20%', '40%', '60%', '80%', '100%'])

results, bin_edges = pd.qcut(lpvCandidates[' cflvsc.Ngoodmeasures'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=['20%', '40%', '60%', '80%', '100%'],
                              retbins = True)

results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                            columns=['Threshold', 'Tier'])

print(results_table)

#literaturePeriods['mod good obs'] = lpvCandidates[' cflvsc.Ngoodmeasures'] // 10
print(literaturePeriods['mod good obs'].unique())
print(literaturePeriods['mod good obs'].value_counts())

for otherColumn in [' cflvsc.ksEMeanMagPawprint']:
    jointplot_w_hue(data=literaturePeriods, x = " cflvsc.Avar", y = otherColumn, hue = 'mod good obs')['fig']

for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName = frequency+' period'
    literaturePeriods[frequencyName] = np.ones(len(literaturePeriods[frequency]))
    literaturePeriods[frequencyName] = literaturePeriods[frequencyName].div(literaturePeriods[frequency])

    logName = frequency + ' log10_period'
    literaturePeriods[logName] = np.log10(literaturePeriods[frequencyName])
    literaturePeriods['crossmatch_1og10_period'] = np.log10(literaturePeriods[' cflvsc.Period'])

    jointplot_w_hue(data=literaturePeriods, x = 'crossmatch_1og10_period', y = logName, hue = 'mod good obs')['fig']
    #sns.jointplot(x='crossmatch_1og10_period', y=logName, data=literaturePeriods, kind='scatter',color=literaturePeriods['mod good obs'], space=0)
    #plt.savefig("Verified Crossmatched Stars/period" + frequency + "comparison.png")

#exit()

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
                if abs(row[frequencyName]-(frequencyMultiple*row[' cflvsc.Period'])) < row[' cflvsc.Period']/10:
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


#plt.show()

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
