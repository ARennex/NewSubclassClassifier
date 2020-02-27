import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from random import sample
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

makePlot = True

sns.set_style('darkgrid')
def jointplot_w_hue(data, x, y, hue=None, colormap = None,
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
    #ax_yhist.title.set_text("Light curve of Object: " + str(data['# sourceID'].iloc[0]) + ". Average magnitude: " + str(np.mean(data['aperMag3'])))
    #plt.title("Light curve of Object: " + str(data['# sourceID'].iloc[0]) + ". Average magnitude: " + str(np.mean(data['aperMag3'])),x=-1.5 , y=1.0, fontsize = 12)
    #plt.gca().invert_yaxis()
    plt.show()
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

gaiaMagHistogram = True
gaiaPeriodComparison = False
periodComparison = False
amplitudeStudy = False
files = ['resultsGaiaVivaBig.csv','resultsGaiaLPVsBig.csv','AllGaia.csv','AllGaiaBright.csv']

###############################
# How bright is Gaia (in VVV) #
###############################

if gaiaMagHistogram == True:

    Gaia = pd.read_csv('1582813002325O-result.csv',sep=',')
    #This file is all Gaia lpvs and their mean G magnitudes
    print(Gaia)
    hist = Gaia['phot_g_mean_mag'].hist(bins = 20)
    plt.title("Distribution of Gaia Mean G Band Mags for LPVs")
    plt.show()
    exit()

    Gaia = pd.read_csv(files[2],skiprows=8,usecols=['ksAperMag3'],sep=',')
    print(Gaia)
    print(len(Gaia[Gaia['ksAperMag3'] < 12]))
    hist = Gaia.hist(bins = 20)
    plt.show()

    del Gaia

    Gaia = pd.read_csv(files[3],skiprows=8,usecols=['ksAperMag3'],sep=',')
    print(Gaia)
    hist = Gaia.hist(bins = 20)
    plt.show()


##########################################
# Comparing Gaia's Periods to literature #
##########################################

if gaiaPeriodComparison == True:

    crossmatchedData = pd.read_csv('cflvsc_r01_crosssources.dat',sep=',')
    #crossmatchedData[' cflvsc.Period']
    crossmatchedData[' sourceID'] = crossmatchedData['#cflvsc.sourceID']
    print(crossmatchedData)
    #print(crossmatchedData.columns)

    vivaGaia = pd.read_csv(files[1],skiprows=8,sep=',')
    vivaGaia['mainPeriod'] = np.ones(len(vivaGaia['frequency']))
    vivaGaia['mainPeriod'] = vivaGaia['mainPeriod'].div(vivaGaia['frequency'])
    print(vivaGaia)

    new_df = pd.concat([vivaGaia, crossmatchedData], axis=1, join='inner')
    #new_df = pd.merge(crossmatchedData, vivaGaia,  how='right', left_on='#cflvsc.sourceID', right_on='# sourceID')
    print(new_df)
    new_df = new_df[new_df['mainPeriod'] > 0]
    new_df = new_df[new_df['mainPeriod'] < 2000]
    new_df = new_df[new_df[' cflvsc.Period'] > 0]
    new_df = new_df[new_df[' cflvsc.Period'] < 2000]
    print(new_df)

    #vivaGaia = vivaGaia[vivaGaia['flagFbias7'] < 5]

    new_df['mod good obs'] = pd.qcut(new_df[' cflvsc.Ngoodmeasures'],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(new_df[' cflvsc.Ngoodmeasures'],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print(results_table)
    print(new_df['mod good obs'].unique())
    print(new_df['mod good obs'].value_counts())

    jointplot_w_hue(data=new_df, x = " cflvsc.Period", y = 'mainPeriod', hue = 'mod good obs')['fig']

########################################
# Comparing Carlos's Periods to Gaia's #
########################################

if periodComparison == True:
    vivaGaia = pd.read_csv(files[0],skiprows=12,sep=',')
    print(vivaGaia)

    vivaGaia = vivaGaia[vivaGaia['flagFbias7'] < 5]

    vivaGaia['mod good obs'] = pd.qcut(vivaGaia['nGoodMeasurements'],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(vivaGaia['nGoodMeasurements'],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print(results_table)
    print(vivaGaia['mod good obs'].unique())
    print(vivaGaia['mod good obs'].value_counts())

    jointplot_w_hue(data=vivaGaia, x = "aVar", y = 'frequency', hue = 'mod good obs')['fig']

    vivaGaia['mainPeriod'] = np.ones(len(vivaGaia['frequency']))
    vivaGaia['mainPeriod'] = vivaGaia['mainPeriod'].div(vivaGaia['frequency'])
    vivaGaia['crossmatchLogPeriod'] = np.log10(vivaGaia['mainPeriod'])
    for frequency in ['freqPKfi2','freqPLfi2','freqLSG','freqPDM','freqSTR','frequency']:
        frequencyName = frequency+' period'
        vivaGaia[frequencyName] = np.ones(len(vivaGaia[frequency]))
        vivaGaia[frequencyName] = vivaGaia[frequencyName].div(vivaGaia[frequency])

        logName = frequency + ' log10_period'
        vivaGaia[logName] = np.log10(vivaGaia[frequencyName])

        jointplot_w_hue(data=vivaGaia, x = 'crossmatchLogPeriod', y = logName, hue = 'mod good obs')['fig']

    #exit()

    vivaGaia = vivaGaia[vivaGaia['nGoodMeasurements'] > 50]

    vivaGaia['mod good obs'] = pd.qcut(vivaGaia['nGoodMeasurements'],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'])
    results, bin_edges = pd.qcut(vivaGaia['nGoodMeasurements'],
                                  q=[0, .2, .4, .6, .8, 1],
                                  labels=['20%', '40%', '60%', '80%', '100%'],
                                  retbins = True)
    results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                columns=['Threshold', 'Tier'])
    print(results_table)
    print(vivaGaia['mod good obs'].unique())
    print(vivaGaia['mod good obs'].value_counts())

    jointplot_w_hue(data=vivaGaia, x = "aVar", y = 'frequency', hue = 'mod good obs')['fig']

    grid = sns.PairGrid(data= vivaGaia,
                        vars = ['freqPKfi2','freqPLfi2','freqLSG','freqPDM','freqSTR','frequency'], height = 2)

    # Map the plots to the locations
    grid = grid.map_upper(plt.scatter, color = 'darkred')
    grid = grid.map_upper(corr)
    grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
    grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'darkred')
    plt.show()

if amplitudeStudy == True:
    bigGaia = pd.read_csv(files[1],skiprows=8,sep=',')
    print(bigGaia)
    bigGaia['period'] = np.ones(len(bigGaia['frequency']))
    bigGaia['period'] = bigGaia['period'].div(bigGaia['frequency'])

    bigGaia = bigGaia[bigGaia['period'] < 10000]
    bigGaia = bigGaia[bigGaia['period'] > 0]

    bigGaia = bigGaia[bigGaia['ksAperMag3'] < 20]
    bigGaia = bigGaia[bigGaia['ksAperMag3'] > 0]

    print(bigGaia)

    sns.jointplot(x="period", y="ksAperMag3", data=bigGaia, kind='kde', space=0)
    #plt.scatter(bigGaia['period'],bigGaia['ksAperMag3'])
    plt.title("Magnitude VS. Period")
    plt.gca().invert_yaxis()
    plt.show()

    vivaGaia = pd.read_csv(files[0],skiprows=12,sep=',')
    print(vivaGaia)
    vivaGaia['period'] = np.ones(len(vivaGaia['frequency']))
    vivaGaia['period'] = vivaGaia['period'].div(vivaGaia['frequency'])

    vivaGaia = vivaGaia[vivaGaia['period'] < 10000]
    vivaGaia = vivaGaia[vivaGaia['period'] > 0]

    vivaGaia = vivaGaia[vivaGaia['ksAperMag3'] < 20]
    vivaGaia = vivaGaia[vivaGaia['ksAperMag3'] > 0]

    print(vivaGaia)

    sns.jointplot(x="period", y="ksAperMag3", data=vivaGaia, kind='kde', space=0)
    #plt.scatter(bigGaia['period'],bigGaia['ksAperMag3'])
    plt.title("Magnitude VS. Period")
    plt.gca().invert_yaxis()
    plt.show()

    bigGaia = bigGaia[bigGaia['ksAperMag3'] > 12]

    print(bigGaia)
