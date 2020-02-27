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

files = ['alt values(1).csv','resultsZTFCrossmatches.csv']

ztfFiles = pd.read_csv(files[0],sep=',')
ztfCrossmatch = pd.read_csv(files[1],skiprows=14,sep=',')

#print(ztfFiles)
#print(ztfCrossmatch)
ztfCrossmatch = ztfCrossmatch[ztfCrossmatch['distance'] > 0.0]
print("Actual Matches: ", ztfCrossmatch)

#new_df = pd.merge(ztfFiles, ztfCrossmatch,  how='left', left_on=['ra','decl'], right_on = ['upload_RA','upload_Dec'])
new_df = pd.merge(ztfCrossmatch, ztfFiles,  how='left', left_on=['upload_RA','upload_Dec'], right_on = ['ra','decl'])
new_df.to_csv('ztfVVVMatched.csv',index=False)
print(new_df)

new_df = pd.concat([ztfCrossmatch, ztfFiles], axis=1, join='inner')
#new_df = pd.merge(ztfCrossmatch, ztfFiles,  how='inner', left_on=['upload_RA','upload_Dec'], right_on = ['ra','decl'])
new_df.to_csv('ztfVVVMatched.csv',index=False)
print(new_df)

just_data = new_df[['sourceID','objectId']]
just_data.to_csv('ztfVVVOnlyRelevantColumns',index=False)
print(just_data)
