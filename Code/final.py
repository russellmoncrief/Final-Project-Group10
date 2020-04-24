#Master Python File

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

#change directory to where images are stored
os.chdir("/home/ubuntu/Machine-Learning/Python-Math/")

#load the x-ray data/targets
xray_data = pd.read_csv("/home/ubuntu/Machine-Learning/Python-Math/Data_Entry_2017.csv")
print("Number of observations:", len(xray_data))
print(xray_data.head(5))

#compile list of directories where images are stored
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

#iterate over directories and compile .png files/ map to x-ray data
for path in dir:
    my_glob = glob(path + "*.png")
    print(len(my_glob))
    full_img_paths = {os.path.basename(x): x for x in my_glob}
    xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)

print(len(xray_data))
num_unique_labels = xray_data['Finding Labels'].nunique()
print('Number of unique labels:',num_unique_labels)

#print frequecy of labels
count_per_unique_label = xray_data['Finding Labels'].value_counts()
df_count_per_unique_label = count_per_unique_label.to_frame() 
print(df_count_per_unique_label)

#graph features and their frequncies
sns.barplot(x = df_count_per_unique_label.index[:20],
y="Finding Labels", data=df_count_per_unique_label[:20], color = "green")
plt.xticks(rotation = 90) # visualize results graphically
plt.show()

#print shape of data
print(xray_data.shape)


