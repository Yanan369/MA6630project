#随机分裂训练集和测试集
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('train.csv')
df.values
train, test = train_test_split(df, test_size=0.3, random_state=0)
train.to_csv("train1.csv",index=False, sep=',')
test.to_csv("test1.csv",index=False, sep=',')