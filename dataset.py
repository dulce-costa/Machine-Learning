import os
import seaborn as sns

os.makedirs("data", exist_ok= "True")
df = sns.load_dataset('penguins')
df.to_csv('data/peng.csv', index= False)