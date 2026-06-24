import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv('train_80.csv', index=False)
test_data.to_csv('test_20.csv', index=False)