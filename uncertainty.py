#Calculate uncertainty statistic of each feature using categorical feature set
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter

#calculate conditional entropy

def conditional_entropy(x, y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * np.log(p_y/p_xy)
    return entropy

def uncertainty(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = entropy(p_x)
    if s_x == 0:
        return 0
    else:
        return (s_x - s_xy) / s_x

df= pd.read_csv('dataset.csv', encoding= 'unicode_escape')
categories = df[df.columns[559:]]
features = df[df.columns[:558]]
for feature in features:
    for category in categories:
      print(f"{uncertainty(df[category], df[feature])} is the correlation between {feature} and {category} ")