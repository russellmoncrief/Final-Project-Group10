import numpy as np
import pandas as pd
import os
from glob import glob
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix


###%% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
LR = 0.02       #1e-3
N_EPOCHS = 2
BATCH_SIZE = 512


data_path = '/home/ubuntu/Deep-Learning/'
data = pd.read_csv(os.path.join(data_path,'Data_Entry_2017.csv'))

image_paths = glob(os.path.join(data_path,'xRays','images','*.png'))

all_image_paths = { os.path.basename(x): x for x in image_paths }

print('Scans found     : ', len(all_image_paths), 'images')
print('Total Headers   : ', data.shape[0], 'headers')
print('Unique Patients : ',len(np.unique(data['Patient ID'])), ' patients')

data['path'] = data['Image Index'].map(all_image_paths.get)




###Here we take the labels and make them into a more clear format.
##The primary step is to see the distribution of findings and then to convert them to simple binary labels.

label_counts = data['Finding Labels'].value_counts()[0:20] # Top 20 combination of Finding Labels
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
_ = ax1.set_ylabel('Count')
plt.show()


ax = plt.figure(figsize=(30, 8))
sns.countplot(data['Patient Age'])
plt.xlabel('Patient Age')
plt.ylabel('Count')
plt.show()


from itertools import chain
all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
data['Finding Labels List'] = data['Finding Labels'].map(lambda x: x.split('|')).tolist()
print('All Pathologies/Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    data[c_label] = data['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)
print(data)


for c_label in all_labels:
    print((c_label,int(data[c_label].sum())))


#Removing irrelevant columns
data.drop(['Image Index','Finding Labels','Follow-up #','OriginalImage[Width','Height]',
                  'OriginalImagePixelSpacing[x','y]','Unnamed: 11'],axis=1, inplace=True)

data['Patient Gender']=data['Patient Gender'].map(lambda gender: 0 if gender=='M' else 1)
data['View Position']=data['View Position'].map(lambda vp: 0 if vp=='AP' else 1)
print(data)



patients=np.unique(data['Patient ID'])
from sklearn.model_selection import train_test_split
train_valid, test = train_test_split(patients,
                                   test_size = 0.2,
                                   random_state = 2020)
train, valid = train_test_split(train_valid,
                                test_size = 0.12,
                                random_state = 2020)
print('No. of Unique Patients in Train dataset : ',len(train))
print('No. of Unique Patients in Valid dataset : ',len(valid))
print('No. of Unique Patients in Test dataset  : ',len(test))

train_df = data[data['Patient ID'].isin(train)]
valid_df = data[data['Patient ID'].isin(valid)]
test_df = data[data['Patient ID'].isin(test)]
print('\nTraining Dataframe   : ', train_df.shape[0],' images')
print('Validation Dataframe : ', valid_df.shape[0],' images')
print('Testing Dataframe    : ', test_df.shape[0],' images')

print("===============================")
print('Count of each label in the Train dataset')
for c_label in all_labels:
    print((c_label,int(train_df[c_label].sum())))

print('\nCount of each label in the Valid dataset')
for c_label in all_labels:
    print((c_label,int(valid_df[c_label].sum())))

print('\nCount of each label in the Test dataset')
for c_label in all_labels:
    print((c_label,int(test_df[c_label].sum())))


pd.options.mode.chained_assignment = None

##Train data

min_age = min(train_df['Patient Age'])
diff = max(train_df['Patient Age']) - min_age
train_df['Patient Age'] = train_df['Patient Age'].map(lambda age: (age-min_age)/diff)
train_df.drop('Patient ID',axis=1,inplace=True)

##Validation data
min_age = min(valid_df['Patient Age'])
diff = max(valid_df['Patient Age']) - min_age
valid_df['Patient Age'] = valid_df['Patient Age'].map(lambda age: (age-min_age)/diff)
valid_df.drop('Patient ID',axis=1,inplace=True)

##Test data
min_age = min(test_df['Patient Age'])
diff = max(test_df['Patient Age']) - min_age
test_df['Patient Age'] = test_df['Patient Age'].map(lambda age: (age-min_age)/diff)
test_df.drop('Patient ID',axis=1,inplace=True)


## Preprocessing

IMG_SIZE = (224, 224)   ## Resize images

data_gen = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip = True,
                              vertical_flip = False,
                              height_shift_range= 0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)



# Creating DataFrame Iterators

print('Generating Training DataFrame Iterator')
train_gen = data_gen.flow_from_dataframe(dataframe=train_df,
                                                directory=None,
                                                x_col = 'path',
                                                y_col = 'Finding Labels List',
                                                target_size = IMG_SIZE,
                                                class_mode='categorical',
                                                color_mode = 'grayscale',
                                                batch_size = 64)

print('\nGenerating Validation DataFrame Iterator')
valid_gen = data_gen.flow_from_dataframe(dataframe=valid_df,
                                                directory=None,
                                                x_col = 'path',
                                                y_col = 'Finding Labels List',
                                                target_size = IMG_SIZE,
                                                color_mode = 'grayscale',
                                                class_mode='categorical',
                                                batch_size = 64)

print('\nGenerating Test DataFrame Iterator')
test_gen = data_gen.flow_from_dataframe(dataframe=test_df,
                                                directory=None,
                                                x_col = 'path',
                                                y_col ='Finding Labels List',
                                                target_size = IMG_SIZE,
                                                color_mode = 'grayscale',
                                                class_mode='categorical',
                                                batch_size = 64)

t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(train_gen.class_indices.keys(), c_y)
                             if n_score>0.5]))
    c_ax.axis('off')
plt.show()

train_gen.reset()
print("^^^^^^^^^^^^^^^^^^^^")


## Model

model = Sequential()
# conv1
model.add(Conv2D(64, kernel_size=(6,6),
                activation='relu',
                border_mode='same',
                input_shape=(224,224,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv2
model.add(Conv2D(64, kernel_size=(2,2),
                 activation='relu',
                 border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# fc
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))  ## initial -  (Dropout(0.5) & no batchNorm
model.add(Dense(len(all_labels), activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
history = model.fit_generator(train_gen,
                              steps_per_epoch=train_gen.n//train_gen.batch_size,
                              validation_data = valid_gen,
                              validation_steps = valid_gen.n//valid_gen.batch_size,
                              epochs = N_EPOCHS)

model.save("CNN.h5")  ## Save model

model = load_model("CNN.h5")  ## Load model for test set evaluation

test_gen.reset()
steps = len(test_gen.classes) // test_gen.batch_size

test_acc_list = []
test_loss_list = []

for i in range(steps):
    test_X, test_Y = next(test_gen)
    test_loss, test_acc = model.evaluate(test_X, test_Y, test_gen.batch_size, verbose=0)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

test_accuracy = np.mean(test_acc_list)
test_loss = np.mean(test_loss_list)
print('Test Accuracy : ', test_accuracy, 'Test Loss : ', test_loss)


fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['acc', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

plt.show()



preds = model.predict(test_X)

acc = accuracy_score(test_Y, np.round(preds))*100
cm = confusion_matrix(test_Y, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('\nTRAIN METRIC ----------------------')
print('Train acc: {}'.format(np.round((history.history['acc'][-1])*100, 2)))






