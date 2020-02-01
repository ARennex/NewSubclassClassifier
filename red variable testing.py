import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

makePlot = True
makePlot2 = True

vvv_data_files = 'results*.csv'
files_collection = np.array(list(glob.iglob(vvv_data_files, recursive=True)))
print("Collecting query results. List is: ", files_collection)
files_collection = ['results1_14_50_23_4618.csv']

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

    freq_clipped = fap_clipped[fap_clipped['freqLSG'] < 2.0]
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

    print("Effect of clips: Original, RMS Clipped, FAP Clipped, Freq Clipped")
    print(len(red_var_data['ksMagRms']),len(rms_clipped['ksMagRms']),len(fap_clipped['ksMagRms']),len(freq_clipped['ksMagRms']))
    #sns.jointplot(x=freq_clipped["faPcorrelation2"], y=freq_clipped["freqLSG"], kind='kde', color="grey", space=0)
    #plt.show()
