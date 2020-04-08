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
import vvvObjectAccess

magAplPlot = False
OSARGCheck = True
mode='period'

ap = argparse.ArgumentParser()
ap.add_argument("-O", "--OSARGCheck", type=bool, default=False,
    help="OSARG plotting y?n?.")
ap.add_argument("-m", "--mode", type=str, default='default',
    help="Mode to plot.")
args = vars(ap.parse_args())
OSARGCheck = args['OSARGCheck']
mode=args['mode']

from pylab import rcParams
rcParams['figure.figsize'] = 14, 10

sns.set_style('darkgrid')
def jointplot_w_hue(data, x, y, filename, plotLims=None, hue=None, colormap = None,
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

    # active_colormap = colormap[0: len(hue_groups)]
    # legend_mapping = []
    # for hue_grp, color in zip(hue_groups, active_colormap):
    #     legend_entry = mpatches.Patch(color=color, label=hue_grp)
    #     legend_mapping.append(legend_entry)
    #
    #     subdata[hue_grp] = data[data[hue]==hue_grp]
    #     colors[hue_grp] = color

    import pylab
    NUM_COLORS = len(hue_groups)
    active_colormap = []
    legend_mapping = []
    cm = pylab.get_cmap('gist_rainbow')
    for i in range(NUM_COLORS):
        active_colormap.append(cm(1.*i/NUM_COLORS))
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
        #print(hue_grp)
        #print(subdata[hue_grp])
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

    if plotLims != None:
        left, right = plotLims[0], plotLims[1]
        ax_main.set_ylim(left-0.1, right+0.1)

    ax_legend.legend(handles=legend_mapping)
    if plotLims != None:
        filename += 'Limited'
    os.makedirs(('Selected Verified Crossmatched Stars/' + filename + '/'), exist_ok=True)
    saveName = "Selected Verified Crossmatched Stars/"+ filename + '/' + x + y + ".png"
    #saveName = "Selected Verified Crossmatched Stars/"+ x + y + filename + ".png"
    plt.savefig(saveName)
    #plt.savefig("Selected Verified Crossmatched Stars/"+ x + y + "FrequencyHeight.png")
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

literaturePeriods = crossmatchedData[crossmatchedData[' cflvsc.Period'] > 0]

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

def selectionCuts(df, mode):
    #Make cuts based on the various quantities
    tempData=df
    del df
    print("Amount of data: ", len(tempData))
    if mode == 'crossmatched':
        for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
            frequencyName,logName,columnName,thresholdName,logHeight = crossmatchedNames(frequency)
            if frequency == ' cflvsc.FreqKfi2':
                tempData = tempData[tempData[thresholdName] > 1.0]
                print("Amount of data: ", len(tempData))
            if frequency == ' cflvsc.FreqLfl2':
                tempData = tempData[tempData[logHeight] > -1.0]
                print("Amount of data: ", len(tempData))
            if frequency == ' cflvsc.FreqPDM':
                tempData = tempData[tempData[logHeight] > -1.0]
                print("Amount of data: ", len(tempData))

    elif mode == 'viva':
        for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
            frequencyName,_,logName,columnName,thresholdName,logHeight = vivaNames(frequency)
            if frequency == ' cflvsc.FreqKfi2':
                tempData = tempData[tempData[thresholdName] > 1.0]
                print("Amount of data: ", len(tempData))
            if frequency == ' cflvsc.FreqLfl2':
                tempData = tempData[tempData[logHeight] > -1.0]
                print("Amount of data: ", len(tempData))
            if frequency == ' cflvsc.FreqPDM':
                tempData = tempData[tempData[logHeight] > -1.0]
                print("Amount of data: ", len(tempData))

    return tempData

def vivaplot():
    if mode == 'default':
        """Plot the viva data"""
        gridsize=60
        temp_data = vivaData[vivaData[vivaThresholdName] < 5]
        temp_data = temp_data[temp_data[vivaThresholdName] >= 0]
        plt.hexbin(temp_data[vivaLogPeriod], temp_data[vivaThresholdName], gridsize=gridsize, cmap=cm.jet, bins='log')
        #plt.hexbin(temp_data[vivaLogPeriod], temp_data[vivaThresholdName], gridsize=gridsize, cmap=cm.jet, bins=None)
        xedges = [temp_data[vivaLogPeriod].min(), temp_data[vivaLogPeriod].max()]
        yedges = [temp_data[vivaThresholdName].min(), temp_data[vivaThresholdName].max()]
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
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.axis(extent)

        cb = plt.colorbar()
        cb.set_label('mean value')
        plt.savefig('Selected Verified Crossmatched Stars/PERIOD CUTS/'+frequencyName+'ORIGINAL.png')
        #plt.show()
        plt.clf()

def columnProcessing(df, mode):
    for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
        if mode == 'crossmatched':
            frequencyName,logName,columnName,thresholdName,logHeight = crossmatchedNames(frequency)
            #frequencyName = frequency+' period'
            df[frequencyName] = np.ones(len(df[frequency]))
            df[frequencyName] = df[frequencyName].div(df[frequency])
            #logName = frequency + ' log10_period'
            df[logName] = np.log10(df[frequencyName])
            df['crossmatch_1og10_period'] = np.log10(df[' cflvsc.Period'])
            #columnName = frequency.replace("Freq",'Height')
            #thresholdName = columnName+'-FAP'
            df[thresholdName] = df[columnName]/df[' cflvsc.FAPcorrelation2']
            df[logHeight] = np.log10(df[columnName])


        elif mode == 'viva':
            vivaName,vivaPeriod,vivaLogPeriod,vivaHeight,vivaThresholdName,vivaLogHeight = vivaNames(frequency)
            df[vivaPeriod] = np.ones(len(df[vivaName]))
            df[vivaPeriod] = df[vivaPeriod].div(df[vivaName])
            #vivaLogPeriod = vivaName+'LogPeriod'
            df[vivaLogPeriod] = np.log10(df[vivaPeriod])
            #vivaHeight = vivaName.replace("freq",'height')
            #vivaThresholdName = vivaHeight+'-FAP'
            df[vivaThresholdName] = df[vivaHeight]/df['faPcorrelation2']
            df[vivaLogHeight] = np.log10(df[vivaHeight])

    return df

def periodAnalysis():
    ##########################################################################
    # Some types of variable objects are visible as clumps in variable space #
    # This section is designed to test this
    ##########################################################################

    ##########################################
    # Read and do default processing on data #
    ##########################################
    vivaData = pd.read_csv('QualityGoodObjsColors.csv', skiprows=20) #This dataset actually has proper magnitudes and colours
    vivaData = columnProcessing(vivaData,'viva')
    vivaData = selectionCuts(vivaData,'viva')

    vivaName,vivaPeriod,vivaLogPeriod,vivaHeight,vivaThresholdName,vivaLogHeight = vivaNames(' cflvsc.FreqLfl2')
    vivaData = vivaData[vivaData[vivaLogHeight] > 1.0]
    print("Amount of data: ", len(vivaData))
    vivaData['jmhPnt'] = vivaData['jAperMag3']-vivaData['hAperMag3']
    vivaData['hmksPnt'] = vivaData['hAperMag3']-vivaData['ksAperMag3']

    vivaName,vivaPeriod,vivaLogPeriod,vivaHeight,vivaThresholdName,vivaLogHeight = vivaNames(' cflvsc.FreqLSG')
    sample = vivaData[vivaData[vivaPeriod] > 10].sample(n=3)

    for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
        vivaName,vivaPeriod,vivaLogPeriod,vivaHeight,vivaThresholdName,vivaLogHeight = vivaNames(frequency)
        for ids,periods in zip(sample['# vivaID'].values,sample[vivaPeriod].values):
            vvvObjectAccess.main(id=ids,phase=periods,save=True)
        # results, bin_edges = pd.qcut(vivaData[vivaHeight],
        #                               q=[0, .2, .4, .6, .8, 1],
        #                               labels=['20%', '40%', '60%', '80%', '100%'],
        #                               retbins = True)
        # vivaData['mod heights'] = pd.qcut(vivaData[vivaHeight],
        #                               q=[0, .2, .4, .6, .8, 1],
        #                               labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
        # results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
        #                             columns=['Threshold', 'Tier'])
        # print('*'*50)
        # print(frequency)
        # print('*'*50)
        # print(results_table)
        # print(vivaData['mod heights'].unique())
        # print(vivaData['mod heights'].value_counts())
        #
        # jointplot_w_hue(vivaData, 'aVar', vivaLogPeriod, filename='amplitude VS period', hue='mod heights')['fig']
        for column in ['jmhPnt','hmksPnt',vivaHeight]:
            tempData = vivaData[vivaData[column] > 0]
            results, bin_edges = pd.qcut(tempData[column],
                                          q=[0, .2, .4, .6, .8, 1],
                                          labels=['20%', '40%', '60%', '80%', '100%'],
                                          retbins = True)

            tempData['mod'] = pd.qcut(tempData[vivaHeight],
                                          q=[0, .2, .4, .6, .8, 1],
                                          labels=bin_edges[:-1]) #-2 in order to drop the last value only. the bin labels are the min point
            results_table = pd.DataFrame(zip(bin_edges, ['20%', '40%', '60%', '80%', '100%']),
                                        columns=['Threshold', 'Tier'])
            print('*'*50)
            print(frequency)
            print('*'*50)
            print(results_table)
            print(tempData['mod'].unique())
            print(tempData['mod'].value_counts())

            jointplot_w_hue(tempData, 'aVar', vivaLogPeriod, filename='amplitude VS period - '+column, hue='mod')['fig']



        temp = vivaData[vivaData['wesenheit'] > 0]
        g = sns.jointplot(vivaLogPeriod, "wesenheit", data=temp, kind="reg")
        filename='wesenheit'
        os.makedirs(('Selected Verified Crossmatched Stars/' + filename + '/'), exist_ok=True)
        saveName = "Selected Verified Crossmatched Stars/"+ filename + '/' + vivaLogPeriod + 'wesenheit' + ".png"
        plt.savefig(saveName)
        plt.close()


periodAnalysis()
exit()

import matplotlib.cm as cm
vivaData = pd.read_csv('AllGoodObjs.csv', skiprows=20)
compilationData = literaturePeriods[literaturePeriods[' cflvsc.Ngoodmeasures'] > 40]
print(compilationData[compilationData[' cflvsc.MainVarType'] == '  EB'])
counts = compilationData[' cflvsc.MainVarType'].value_counts()
valid = counts[counts > 40]
print(counts)
print(valid)
compilationData = compilationData[compilationData[' cflvsc.MainVarType'].isin(valid.index.values)]

compilationData = columnProcessing(compilationData,'crossmatched')
vivaData = columnProcessing(vivaData,'viva')

compilationData = selectionCuts(compilationData,'crossmatched')
vivaData = selectionCuts(vivaData,'viva')

for frequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
    frequencyName,logName,columnName,thresholdName,logHeight = crossmatchedNames(frequency)
    vivaName,vivaPeriod,vivaLogPeriod,vivaHeight,vivaThresholdName,vivaLogHeight = vivaNames(frequency)


    if mode == 'period':
        for xCoordFrequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
            _,_,xVivaName,_,_,_ = vivaNames(xCoordFrequency) #Get the needed vivaLogPeriod name
            _,xName,_,_,_ = crossmatchedNames(xCoordFrequency)

            """Plot the viva data"""
            gridsize=60
            temp_data = vivaData[vivaData[vivaThresholdName] < 5]
            temp_data = temp_data[temp_data[vivaThresholdName] >= 0]
            plt.hexbin(temp_data[xVivaName], temp_data[vivaThresholdName], gridsize=gridsize, cmap=cm.jet, bins='log')
            #plt.hexbin(temp_data[vivaLogPeriod], temp_data[vivaThresholdName], gridsize=gridsize, cmap=cm.jet, bins=None)
            xedges = [temp_data[xVivaName].min(), temp_data[xVivaName].max()]
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
            plt.xlabel(xVivaName + ' & crossmatched period')

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

            #plt.hexbin(x,y, bins='log', cmap=dropout_high)

            #xName = 'crossmatch_1og10_period'
            #xName = logName
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
            #plt.title(vivaName)
            #plt.axis([compilationData['crossmatch_1og10_period'].min(), compilationData['crossmatch_1og10_period'].max(), compilationData[thresholdName].min(), compilationData[thresholdName].max()])

            cb = plt.colorbar()
            cb.set_label('Counts')
            plt.savefig('Selected Verified Crossmatched Stars/PERIOD CUTS/'+thresholdName+'VS'+xVivaName+'.png')
            #plt.show()
            plt.clf()

    if mode == 'height':
        os.makedirs(('Selected Verified Crossmatched Stars/LOG HEIGHT CUTS/'), exist_ok=True)
        for xCoordFrequency in [' cflvsc.FreqSTR',' cflvsc.FreqPDM',' cflvsc.FreqLSG',' cflvsc.FreqKfi2',' cflvsc.FreqLfl2']:
            _,_,_,_,_,xvivaLogHeight = vivaNames(xCoordFrequency) #Get the needed names for the graph
            _,_,_,_,xlogHeight = crossmatchedNames(xCoordFrequency)
            print("Now on: ", frequency, ', ' + xCoordFrequency)

            temp_data1 = vivaData[[xvivaLogHeight,vivaThresholdName]]
            temp_data1.columns = ['x1','y1']

            temp_data2 = compilationData[[xlogHeight,thresholdName]]
            temp_data2.columns = ['x2','y2']
            #print(temp_data1.describe())
            #print(temp_data2.describe())

            graph = sns.jointplot(x=temp_data1.x1, y=temp_data1.y1, color='r')
            graph.x = temp_data2.x2
            graph.y = temp_data2.y2
            graph.plot_joint(plt.scatter, marker='x', c='b')
            #graph.plot_joint(plt.scatter, marker='x', c='b', s=50)
            # print(frequency+'VS'+xCoordFrequency)
            # print(max(temp_data2.x2), ', ', max(temp_data2.y2))
            # print(max(compilationData[logHeight]), ', ', max(compilationData[thresholdName]))
            #plt.scatter(temp_data2.x2, y=temp_data2.y2)
            #plt.title('Scatter plot of power spectrum heights and heights/faPcorrelation2')
            plt.ylabel(vivaThresholdName)
            plt.xlabel(xvivaLogHeight)
            #plt.yscale('log')
            plt.savefig('Selected Verified Crossmatched Stars/LOG HEIGHT CUTS/'+xCoordFrequency +'VS'+frequency+'.png')
            #plt.show()
            plt.close()
            plt.clf()

            categories = pd.unique(compilationData[' cflvsc.MainVarType'])
            print(categories)
            categories = list(np.sort(categories))
            print(categories)
            B = categories[:len(categories)//2]
            C = categories[len(categories)//2:]
            counter = 0
            for section in [B,C]:
                tempData = compilationData[compilationData[' cflvsc.MainVarType'].isin(section)]
                jointplot_w_hue(tempData, xlogHeight, thresholdName, filename='colouredHeights'+str(counter), hue=' cflvsc.MainVarType')['fig']
                counter += 1

            # if xCoordFrequency == ' cflvsc.FreqLfl2':
            #     grid = sns.jointplot(x="x2", y="y2", data=temp_data2, kind='scatter')
            #     plt.savefig('Selected Verified Crossmatched Stars/LOG HEIGHT CUTS/'+frequency+'.png')
            #     plt.clf()
            #
            #     grid = sns.jointplot(x=temp_data2.x2, y=compilationData[thresholdName], kind='scatter')
            #     plt.savefig('Selected Verified Crossmatched Stars/LOG HEIGHT CUTS/test'+frequency+'.png')
            #     plt.clf()


            del temp_data1, temp_data2, graph

            #grid = sns.jointplot(x=" cflvsc.Avar", y=" cflvsc.ksEMeanMagPawprint", data=temp_data, kind='kde')
            #quit()
