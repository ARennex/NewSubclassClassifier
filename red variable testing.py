import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

makePlot = False
makePlot2 = False
makePlot3 = False
makePlot4 = True

# vvv_data_files = 'results*.csv'
# files_collection = np.array(list(glob.iglob(vvv_data_files, recursive=True)))
# print("Collecting query results. List is: ", files_collection)
base_path = os.getcwd()
from pathlib import Path
p = Path(base_path)
one_up = str(p.parent)
print("Moving one level up to: ", str(one_up))
files = one_up + '/results1_14_50_23_4618.csv'
files_collection = [files]
#files_collection = ['results1_14_50_23_4618.csv']

for vvv_file in files_collection:
    red_var_data = pd.read_csv(vvv_file,skiprows=12,sep=',')
    rms_clipped = red_var_data[red_var_data['ksMagRms'] > 0]
    fap_clipped = rms_clipped[rms_clipped['faPcorrelation2'] > 0]
    labels = ['ksMagRms','faPcorrelation2','freqLSG','ksAmpl','jmhPnt','wesenheit']
    for variable in labels:
        print(variable + "original mean: ", np.mean(red_var_data[variable]))
        print(variable + "rms clipped mean: ", np.mean(rms_clipped[variable]))
        print(variable + "fap clipped mean: ",np.mean(fap_clipped[variable]))
        if makePlot:
            if variable == 'ksMagRms':
                #fap_clipped.plot.scatter(x='faPcorrelation2', y='freqLSG')
                sns.jointplot(x=fap_clipped["faPcorrelation2"], y=fap_clipped["freqLSG"], kind='kde', color="grey", space=0)
                plt.savefig("fapVSfreq.png")
            else:
                #fap_clipped.plot.scatter(x='ksMagRms', y=variable)
                sns.jointplot(x=fap_clipped["ksMagRms"], y=fap_clipped[variable], kind='kde', color="grey", space=0)
                plt.savefig("rmsVS"+variable+'.png')
            plt.clf()
            #plt.show()

    wes_clipped = fap_clipped[fap_clipped['wesenheit'] < 100]
    wes_clipped = wes_clipped[wes_clipped['wesenheit'] > -100]
    wes_clipped['period'] = np.ones(len(wes_clipped['ksMagRms']))
    wes_clipped['period'] = wes_clipped['period'].div(wes_clipped['freqLSG'])
    freq_clipped = wes_clipped[wes_clipped['freqLSG'] < 2.0]
    os.makedirs(('Long Period Red Stars/'), exist_ok=True)
    for variable in labels:
        print(variable + " freq clipped mean: ", np.mean(freq_clipped[variable]))
        if makePlot2:
            if variable == 'ksMagRms':
                #fap_clipped.plot.scatter(x='faPcorrelation2', y='freqLSG')
                sns.jointplot(x=freq_clipped["faPcorrelation2"], y=freq_clipped["freqLSG"], kind='kde', color="grey", space=0)
                plt.savefig('Long Period Red Stars/fap2VSfreq.png')
            else:
                #fap_clipped.plot.scatter(x='ksMagRms', y=variable)
                sns.jointplot(x=freq_clipped["ksMagRms"], y=freq_clipped[variable], kind='kde', color="grey", space=0)
                plt.savefig('Long Period Red Stars/rmsVS'+variable+'.png')
            plt.clf()

    print(wes_clipped.describe())
    print("Effect of clips: Original, RMS Clipped, FAP Clipped, Wes Clipped, Freq Clipped")
    print(len(red_var_data['ksMagRms']),len(rms_clipped['ksMagRms']),len(fap_clipped['ksMagRms']),len(wes_clipped['ksMagRms']),len(freq_clipped['ksMagRms']))
    if makePlot3:
        sns.jointplot(x=wes_clipped["period"], y=wes_clipped["wesenheit"], kind='kde', color="grey", space=0)
        plt.savefig('Long Period Red Stars/WesenheitVSPeriod.png')
        plt.clf()
    per_clipped = wes_clipped[wes_clipped['period'] > 1]
    per_clipped = per_clipped[per_clipped['wesenheit'] > 7.5]
    per_clipped = per_clipped[per_clipped['wesenheit'] < 15.5]
    per_clipped['log10_period'] = np.log10(per_clipped['period'])
    print(per_clipped)
    if makePlot4:
        g = sns.jointplot(x=per_clipped["log10_period"], y=per_clipped["wesenheit"], kind='kde', color="grey", space=0)
        #g.ax_joint.set_xscale('log')
        #plt.xscale('log')
        #plt.gca().invert_yaxis()
        g.ax_joint.invert_yaxis()
        g.savefig('Long Period Red Stars/WesenheitVSPeriod Focused.png')

        # g = sns.jointplot(x=per_clipped["period"], y=per_clipped["wesenheit"], kind='scatter', color="grey")
        # g.ax_joint.set_xscale('log')
        # g.ax_joint.invert_yaxis()
        # plt.show()

        g = sns.JointGrid('period', 'wesenheit', per_clipped)
        #g.plot_marginals(sns.distplot, hist=True, kde=True, color='blue',bins=mybins)
        g.plot_joint(plt.scatter, color='black', edgecolor='black')
        ax = g.ax_joint
        ax.set_xscale('log')
        ax.set_yscale('log')
        g.ax_marg_x.set_xscale('log')
        g.ax_marg_y.set_yscale('log')
        g.savefig('Long Period Red Stars/WesenheitVSPeriod Focused 2.png')
