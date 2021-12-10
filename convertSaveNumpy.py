from glob import glob

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

train_files = []
mask_files = glob('kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask', ''))

# print(train_files[:10])
# print(mask_files[:10])


df = pd.DataFrame(data={"filename": train_files, 'mask': mask_files})
df_train, df_test = train_test_split(df, test_size=0.1)
df_train, df_val = train_test_split(df_train, test_size=0.2)

print(df.describe())

# print(df_train.values.shape)  # 2828
# print(df_val.values.shape)  # 708
# print(df_test.values.shape)  # 393

def convert_tiff_to_numpy(x_tiff, y_tiff, x_array, y_array):
    for img, mask in zip(x_tiff, y_tiff):
        image = plt.imread(img)
        mask = plt.imread(mask)
        if x_array.size == 0:
            x_array = numpy.array([image])
        else:
            x_array = np.append(x_array, numpy.array([image]), axis=0)
        if y_array.size == 0:
            y_array = numpy.array([mask])
        else:
            y_array = np.append(y_array, numpy.array([mask]), axis=0)
    return x_array, y_array


x_trainTiff = df_train['filename'].to_numpy()
y_trainTiff = df_train['mask'].to_numpy()

x_testTiff = df_test['filename'].to_numpy()
y_testTiff = df_test['mask'].to_numpy()

x_valTiff = df_val['filename'].to_numpy()
y_valTiff = df_val['mask'].to_numpy()

x_train = np.array([])
y_train = np.array([])

x_test = np.array([])
y_test = np.array([])

x_val = np.array([])
y_val = np.array([])

print("Intiate conversion...")
resultTrain = convert_tiff_to_numpy(x_trainTiff, y_trainTiff, x_train, y_train)
x_train = resultTrain[0]
y_train = resultTrain[1]

resultTest = convert_tiff_to_numpy(x_testTiff, y_testTiff, x_test, y_test)
x_test = resultTest[0]
y_test = resultTest[1]

resultVal = convert_tiff_to_numpy(x_valTiff, y_valTiff, x_val, y_val)
x_val = resultVal[0]
y_val = resultVal[1]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

print(x_val.shape)
print(y_val.shape)

print("Saving...")

np.save("x_train", x_train)
np.save("y_train", y_train)

np.save("x_test", x_test)
np.save("y_test", y_test)

np.save("x_val", x_val)
np.save("y_val", y_val)
