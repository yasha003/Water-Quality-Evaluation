import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("water_potability.csv")
print(df.head())

df.columns

df.describe()

df.info()

# portability 1 - Water safe for drinking & 0 - unsafe for drinking
df.isnull().sum()

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull())  # White bars indicate the null values

df.isnull().sum()

df["ph"] = df["ph"].fillna(df["ph"].mean())
df["Sulfate"] = df["Sulfate"].fillna(df["Sulfate"].mean())
df["Trihalomethanes"] = df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())

x = df.drop("Potability", axis=1)  # Features- All variable expect potability
y = df['Potability']  # Target

x.shape, y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
x_train.shape, x_test.shape

# Random Forest Classifier
model_rfc = RandomForestClassifier()

# Training Model
model_rfc.fit(x_train, y_train)

# Making Prediction
pred_rfc = model_rfc.predict(x_test)

accuracy_score_rfc = accuracy_score(y_test, pred_rfc)
accuracy_score_rfc * 100

cm = confusion_matrix(y_test, pred_rfc)
print(cm)
sns.heatmap(cm,
            annot=True,
            fmt='d')
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

input_data = (
    8.635848719, 203.3615226, 13672.09176
    , 4.563008686
    , 303.3097712
    , 474.6076449
    , 12.3638167
    , 62.79830896
    , 4.401424715
)
# Changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model_rfc.predict(input_data_reshaped)

if (prediction[0] == 1):
    print("Water Quality is Good")
else:
    print("Water Quality is Bad")

import pickle

pickle.dump(model_rfc, open('models/model_rfc.pkl', 'wb'))

model_rfc = pickle.load(open('models/model_rfc.pkl', 'rb'))
print(model_rfc.predict(input_data_reshaped))
