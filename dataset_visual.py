import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler

import data_processing

label_train = data_processing.read_label('data/label_train.csv')[:, 1]
label_test = data_processing.read_label('data/label_test.csv')[:, 1]

labels, counts = np.unique(label_train, return_counts=True)

for i in range(len(labels)):
    plt.bar(labels[i],counts[i])

plt.show()