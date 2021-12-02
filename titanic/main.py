import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_df = pd.read_csv("csv/train.csv", header=0)

# Convert "Sex" to be a dummy variable (female = 0, Male = 1)
train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
train_df.head(3)

# Complement the missing values of "Age" column with average of "Age"
median_age = train_df["Age"].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
  train_df.loc[(train_df.Age.isnull()), "Age"] = median_age

# remove un-used columns
train_df = train_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
train_df.head(3)

# Load test data, Convert "Sex" to be a dummy variable
test_df = pd.read_csv("csv/test.csv", header=0)
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)

# Complement the missing values of "Age" column with average of "Age"
median_age = test_df["Age"].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
  test_df.loc[(test_df.Age.isnull()), "Age"] = median_age

# Copy test data's "PassengerId" column, and remove un-used columns
ids = test_df["PassengerId"].values
test_df = test_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
test_df.head(3)

# Predict with "Random Forest"
train_data = train_df.values
test_data = test_df.values
model = RandomForestClassifier(n_estimators=100)
output = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data).astype(int)

# export result to be "titanic_submit.csv"
submit_file = open("csv/titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, output))
submit_file.close()
