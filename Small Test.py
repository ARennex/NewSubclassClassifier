import pandas as pd
import numpy as np

read_in_data = np.load('ResultsSubclasses/tanh/1) Red 500.npy')
y_actu = pd.Series(read_in_data[0], name='Actual')
y_pred = pd.Series(read_in_data[1], name='Predicted')
s_actu = pd.Series(read_in_data[2], name='Survey')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
print(df_conf_norm)
df_confusion = pd.crosstab([s_actu,y_actu], y_pred, rownames=['Survey','Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)

# from pandas_ml import ConfusionMatrix
# cm = ConfusionMatrix(y_actu, y_pred)
# data = cm.print_stats()
# np.savetxt('ResultsSubclasses/Pandas ML File.txt', data)
