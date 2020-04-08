import numpy as np
import pandas as pd
import argparse
import os.path
import matplotlib.pyplot as plt
import pyvo as vo


# from pyvo.dal import TAPService
# tap = TAPService("http://tap.roe.ac.uk/vsa")
# tap.search("SELECT TOP 10 * FROM VVVDR4.vvvSource")


ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", type=bool, default=False,
    help="Load existing file: True or False.")
ap.add_argument("-id", "--sourceID", type=int, default=515513755025,
    help="sourceID to use.")
ap.add_argument("-list", "--list", type=str, default='nothing',
    help="If searching a list of sourceIDs, file location.")
ap.add_argument("-m", "--mode", type=str, default='plot',
    help="What to do with the object data, once obtained.")
ap.add_argument("-ph", "--phase", type=str, default='nothing',
    help="Plot the object using phase instead of period (requires known period). Can accept a single phase or a location to a list of phases.")
ap.add_argument("-s", "--save", type=bool, default=False,
    help="Save plot, True or False.")
args = vars(ap.parse_args())

def display(voPandas,singlePeriod,save):

    if singlePeriod == 0.: #No period given
        #Colour each and label each filter seperately
        cm = ['red','orange','green','blue','purple']
        filterDict = ['Z','Y','J','H','Ks']

        plt.figure(figsize=(15, 10))

        #Loop and plot for each filter
        for singleFilter in [1,2,3,4,5]:
            singleFilterData = voPandas[voPandas['filterID'] == singleFilter]
            plt.errorbar(x=singleFilterData["mjd"], y=singleFilterData["aperMag3"],
                         yerr=singleFilterData['aperMag3Err'], color=cm[singleFilter-1],
                         label=filterDict[singleFilter-1],linestyle='',marker='o',markersize=2)
        plt.title('Object ID: ' + str(voPandas['sourceID'][0]) + ' light curve')
        plt.xlabel('mjd')
        plt.ylabel('aperMag3')
        plt.gca().invert_yaxis()
        plt.legend()
        if save == True:
            plt.savefig(str(voPandas['sourceID'][0]) + '.png')
            plt.clf()
        else:
            plt.show()
    else:
        voPandas['phase'] = voPandas['mjd'].mod(singlePeriod)
        voPandas['phase'] = voPandas['phase'].div(singlePeriod)
        #Colour each and label each filter seperately
        cm = ['red','orange','green','blue','purple']
        filterDict = ['Z','Y','J','H','Ks']

        plt.figure(figsize=(15, 10))

        #Loop and plot for each filter
        for singleFilter in [1,2,3,4,5]:
            singleFilterData = voPandas[voPandas['filterID'] == singleFilter]
            plt.errorbar(x=singleFilterData["phase"], y=singleFilterData["aperMag3"],
                         yerr=singleFilterData['aperMag3Err'], color=cm[singleFilter-1],
                         label=filterDict[singleFilter-1],linestyle='',marker='o',markersize=2)
        plt.title('Object ID: ' + str(voPandas['sourceID'][0]) + ' light curve. Period: ' + str(singlePeriod))
        plt.xlabel('phase')
        plt.ylabel('aperMag3')
        plt.gca().invert_yaxis()
        plt.legend()
        if save == True:
            plt.savefig(str(voPandas['sourceID'][0]) + 'period ' + str(singlePeriod) + '.png')
            plt.clf()
        else:
            plt.show()


def main(id, load=False, list='nothing', mode='plot', phase='nothing', save=False):
    #If the script has been given a list of inputs for objects
    if list != 'nothing':
        #Read list of input ids from the provided file
        idData = pd.read_csv(list,header=None)
        try:
            idList = idData[0].values
        except Exception as e:
            print("Input file is in the wrong format!")
            exit()
            raise
    #If the script HAS been given a sinle input
    elif list == 'nothing':
        idList = [id]


    #If the script has been given a list of inputs for periods
    floatCheck = False
    try:
        periods = [float(phase)]
        floatCheck = True
    except Exception as e:
        raise
    if floatCheck != True:
        if phase != 'nothing':
            #Read list of input ids from the provided file
            idData = pd.read_csv(list,header=None)
            try:
                periods = idData[0].values
            except Exception as e:
                print("Input file is in the wrong format!")
                exit()
                raise
        #If the script HAS been given a sinle input
        elif phase == 'nothing':
            periods = [0.]

    #if lengths don't match
    if len(idList) != len(periods):
        if len(idList) < len(periods):
            periods = periods[0:len(idList)]
        elif len(idList) > len(periods):
            periods = period.extend([0.]*(len(idList) - len(periods)))

    zippedLists = zip(idList,periods)

    for singleID,singlePeriod in zippedLists:
        if load == False:
            #service = vo.dal.TAPService("http://tap.roe.ac.uk/firethorn/tap/60")
            service = vo.dal.TAPService("http://tap.roe.ac.uk/vsa")
            #print(service.describe())

            query_text = """
            SELECT b.sourceID,d.mjd,d.filterID,d.aperMag3,d.aperMag3Err,d.ppErrBits,b.flag
            FROM VVVDR4..vvvSourceXDetectionBestMatch as b,VVVDR4..vvvDetection as d
            WHERE b.sourceID=""" + str(singleID) + """and b.multiframeID=d.multiframeID and b.extNum=d.extNum and b.seqNum=d.seqNum and d.seqNum>0
            """
            print(query_text)

            #Submit the query
            resultset = service.search(query_text)
            print(resultset)
            #Convert the resulting data to pandas - skip id if nothing is returned
            voPandas = pd.DataFrame(resultset)
            if len(voPandas) == 0:
                print("ID: " + str(singleID) + " returned no rows!")
                continue

            """
            Removed as of version 2, changes in the data retrieved seem to have fixed the need for a correction
            """
            # #Convert the filter, flag and errbits to a readable format
            # print(voPandas['filterID'],voPandas['flag'],voPandas['ppErrBits'])
            # tempFilter,tempFlag,tempBits = [],[],[]
            # for index, row in voPandas.iterrows():
            #     tempFilter.append(row['filterID'][3])
            #     tempFlag.append(row['flag'][3])
            #     tempBits.append(row['ppErrBits'])
            # voPandas['filterID'] = tempFilter
            # voPandas['flag'] = tempFlag
            # voPandas['ppErrBits'] = tempBits

            #filter out bad flagged data
            voPandas = voPandas[voPandas['flag'] == 0]

            #Remove/fix bad magnitudes and errors
            voPandas = voPandas[voPandas['aperMag3'] > 0]
            maxError = max(voPandas['aperMag3Err'].values)
            voPandas.loc[voPandas['aperMag3Err'] < 0,'aperMag3Err'] = maxError*3.5

            #Sort in order of filter and data
            voPandas.sort_values(['filterID','mjd'],inplace=True)
            print(voPandas)

            #Save data
            newFilename = str(singleID) + "mjdData.csv"
            voPandas.to_csv(newFilename, index=False)

        elif load == True:
            #If told to load data from file, find file based on object ID
            newFilename = str(singleID) + "mjdData.csv"
            #If file is not found, it is skipped
            voPandas = pd.read_csv(newFilename)

        if mode == 'plot':
            display(voPandas,singlePeriod,save)

        if 'limitplot' in mode:
            #If only wanting to plot some of the files, pass a limit
            #Limit comes in the form "limitplot3" where 3 is how many plots you want
            limit = mode[9:]
            try:
                limit = int(limit)
            except Exception as e:
                print("Invalid Limit! ", print(limit))
                limit = 1
            i, = np.where(idList == singleID)
            if i < limit:
                display(voPandas,singlePeriod,save)


if __name__ == "__main__":
    #excucute only if run as a script
    main(args["sourceID"],load=args["load"],list=args["list"],mode=args["mode"],phase=args["phase"],save=args["save"])
