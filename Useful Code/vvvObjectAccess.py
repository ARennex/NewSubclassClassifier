import numpy as np
import pandas as pd
import argparse
import os.path
import matplotlib.pyplot as plt
import pyvo as vo

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", type=int, default=0,
    help="Load existing file: 1 for yes, 0 for no.")
ap.add_argument("-id", "--sourceID", type=int, default=515513755025,
    help="sourceID to use.")
ap.add_argument("-list", "--list", type=str, default='nothing',
    help="If searching a list of sourceIDs, file location.")
ap.add_argument("-m", "--mode", type=str, default='plot',
    help="What to do with the object data, once obtained.")
args = vars(ap.parse_args())

def display(voPandas):
    #Colour each and label each filtet seperately
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
    plt.show()

#If the script has been given a list of inputs
if args["list"] != 'nothing':
    #Read list of input ids from the provided file
    idData = pd.read_csv(args["list"],header=None)
    try:
        idList = idData[0].values
    except Exception as e:
        print("Input file is in the wrong format!")
        exit()
        raise
#If the script HAS been given a sinle input
elif args["list"] == 'nothing':
    idList = [args["sourceID"]]

for singleID in idList:
    if args["load"] == 0:
        service = vo.dal.TAPService("http://tap.roe.ac.uk/firethorn/tap/60")
        #print(service.describe())

        query_text = """
        SELECT b.sourceID,d.mjd,d.filterID,d.aperMag3,d.aperMag3Err,d.ppErrBits,b.flag
        FROM VVVDR4.vvvSourceXDetectionBestMatch as b,VVVDR4.vvvDetection as d
        WHERE b.sourceID=""" + str(singleID) + """and b.multiframeID=d.multiframeID and b.extNum=d.extNum and b.seqNum=d.seqNum and d.seqNum>0
        """

        #Submit the query
        resultset = service.search(query_text)
        print(resultset)
        #Convert the resulting data to pandas - skip id if nothing is returned
        voPandas = pd.DataFrame(resultset)
        if len(voPandas) == 0:
            print("ID: " + str(singleID) + " returned no rows!")
            continue

        #Convert the filter, flag and errbits to a readable format
        tempFilter,tempFlag,tempBits = [],[],[]
        for index, row in voPandas.iterrows():
            tempFilter.append(row['filterID'][3])
            tempFlag.append(row['flag'][3])
            tempBits.append(row['ppErrBits'])
        voPandas['filterID'] = tempFilter
        voPandas['flag'] = tempFlag
        voPandas['ppErrBits'] = tempBits

        #filter out bad flagged data
        voPandas = voPandas[voPandas['flag'] == 0]

        #Sort in order of filter and data
        voPandas.sort_values(['filterID','mjd'],inplace=True)
        print(voPandas)

        #Save data
        newFilename = str(singleID) + "mjdData.csv"
        voPandas.to_csv(newFilename, index=False)

    elif args["load"] == 1:
        #If told to load data from file, find file based on object ID
        newFilename = str(singleID) + "mjdData.csv"
        #If file is not found, it is skipped
        voPandas = pd.read_csv(newFilename)

    if args["mode"] == 'plot':
        display(voPandas)

    if 'limitplot' in args["mode"]:
        #If only wanting to plot some of the files, pass a limit
        #Limit comes in the form "limitplot3" where 3 is how many plots you want
        limit = args["mode"][9:]
        try:
            limit = int(limit)
        except Exception as e:
            print("Invalid Limit! ", print(limit))
            limit = 1
        i, = np.where(idList == singleID)
        if i < limit:
            display(voPandas)
