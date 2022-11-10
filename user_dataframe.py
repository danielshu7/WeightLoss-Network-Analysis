import pandas as pd

df1 = pd.read_csv("WeightLoss/users_consolidated.csv", usecols = ['id','height', 'weight', 'bmi'])
df2 = pd.read_csv("WeightLoss/userprofile.csv", usecols = ['user_id','age'])
df1 = df1.rename(columns={"id": "user_id", "height" : "height", "weight" : "weight", "bmi" : "bmi"})
print(df1.head(3))
print(df2.head(3))
user_info = pd.merge(df1, df2, how="outer")
print(user_info.head(3))

