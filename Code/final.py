import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.callbacks import ModelCheckpoint


os.chdir("/home/ubuntu/Machine-Learning/Python-Math/")

#load file that contains features

data = pd.read_csv("/home/ubuntu/Machine-Learning/Python-Math/Data_Entry_2017.csv")
print("Number of observations:", len(data))
print(data.head(5))

#display a sample of images

#image of Cardiomegaly
image1 = mpimg.imread('/home/ubuntu/Machine-Learning/Python-Math/images_001/images/00000001_000.png')
#Image of Cardiomegaly/Emphysema
image2 = mpimg.imread('/home/ubuntu/Machine-Learning/Python-Math/images_001/images/00000001_001.png')
#Image of Cardiomegaly/Effusion
image3 = mpimg.imread('/home/ubuntu/Machine-Learning/Python-Math/images_001/images/00000001_002.png')
plt.imshow(image1)
plt.show()
plt.imshow(image2)
plt.show()
plt.imshow(image3)
plt.show()

#director where jpgs are stored

dir = ["/home/ubuntu/Machine-Learning/Python-Math/images_001/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_002/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_003/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_004/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_005/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_006/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_007/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_008/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_009/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_010/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_011/images/",
       "/home/ubuntu/Machine-Learning/Python-Math/images_012/images/"]

#compile images

my_glob = glob("/home/ubuntu/Machine-Learning/Python-Math/images*/images*/*.png")
print(len(my_glob))
full_img_paths = {os.path.basename(x): x for x in my_glob}
data['full_path'] = data['Image Index'].map(full_img_paths.get)
print(data['full_path'])

#print length of data and unique labels
print(len(data))
num_unique_labels = data['Finding Labels'].nunique()
print('Number of unique labels:',num_unique_labels)

count_per_unique_label = data['Finding Labels'].value_counts()
df_count_per_unique_label = count_per_unique_label.to_frame()
print(df_count_per_unique_label)

#plot unique label counts
sns.barplot(x = df_count_per_unique_label.index[:20],
y="Finding Labels", data=df_count_per_unique_label[:20], color = "green")
plt.xticks(rotation = 90)
plt.show()

#get shape of data
print(data.shape)

# One-hot encode labels
dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

for label in dummy_labels:
    data[label] = data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
print(data.head(20))

#check number of cases per label
label_sum = data[dummy_labels].sum()
print(label_sum)


data['target_vector'] = data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])
print(data.head())

train, test = train_test_split(data, test_size = 0.20, random_state = 0)

# quick check to see that the training and test set were split properly
print('training set - # of observations: ', len(train))
print('test set - # of observations): ', len(test))
print('prior, full data set - # of observations): ', len(data))

data_gen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,
        rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,
        horizontal_flip=True)


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

image_size = (128, 128)
train_gen = flow_from_dataframe(data_gen, train, path_col = 'full_path', y_col = 'target_vector',
target_size = image_size, color_mode = 'grayscale', batch_size = 32)

valid_gen = flow_from_dataframe(data_gen, test, path_col = 'full_path', y_col = 'target_vector',
target_size = image_size, color_mode = 'grayscale', batch_size = 128)

# define test sets
x_test, y_test = next(flow_from_dataframe(data_gen, test, path_col = 'full_path', y_col = 'target_vector',
target_size = image_size, color_mode = 'grayscale', batch_size = 2048))


LR = 1e-3
model = Sequential([Dense(12, input_shape=x_test.shape[1:], activation="relu"), Dense(14, activation="softmax")])

model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

checkpointer = ModelCheckpoint(filepath='weights.best.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only = True)
callbacks_list = [checkpointer]

model.fit_generator(generator = train_gen, steps_per_epoch = 20, epochs = 100, callbacks = callbacks_list,
                    validation_data = (x_test, y_test))

pred= model.predict(x_test, batch_size = 64, verbose = 1)



