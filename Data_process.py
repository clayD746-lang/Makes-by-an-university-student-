import pandas as pd
from matplotlib import pyplot as plt
import io

from tensorflow.python.ops.numpy_ops.np_random import random

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

## Author ClayD in SH.Sumhs at 2025/8/11
# 一个简单的数据处理，简单看一下数据的各项指标后，画散点图发现有异常值，写函数定位并最终确认为周四的指标有问题


dataframe = pd.read_csv(r'C:\Users\1\Desktop\dataset.csv')
print(dataframe.describe())

def plot_the_dataset(feature,label,number_of_points_to_plot):
    plt.xlabel(feature)
    plt.ylabel(label)

    random_examples = dataframe.sample(n = number_of_points_to_plot)
    plt.scatter(random_examples[feature],random_examples[label])
    plt.show()

def plot_a_contiguous_portion_of_dataset(feature,label,start,end):
    plt.xlabel(feature+'day')
    plt.ylabel(label)
    plt.scatter(dataframe[feature][start:end],dataframe[label][start:end])
    plt.show()

for i in range(0,7):
    start = i *50
    end = start + 49
    plot_a_contiguous_portion_of_dataset('calories','test_score',start,end)

thursday_calories = 0
non_thursday_calories = 0
count = 0

for week in range(0,4):
    for day in range(0,7):
        for subject in range(0,50):
            position = (week * 350) + (day * 50) +subject
            if (day == 4):
                thursday_calories += dataframe['calories'][position]
            else:
                count += 1
                non_thursday_calories += dataframe['calories'][position]

mean_of_thursday_calories = thursday_calories/200
mean_of_non_thursday_calories = non_thursday_calories/1200

print(f'周四的卡路里平均值是:  {mean_of_thursday_calories:2f}')
print(f'非周四的卡路里平均值是: {mean_of_non_thursday_calories:2f}')

