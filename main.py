import joblib
import pandas as pd

# Load dataset
df = pd.read_csv("data/telco_customer_churn.csv")

# Check missing values
print(df.isnull().sum())

import pandas as pd

df = pd.read_csv("data/telco_customer_churn.csv")

print(df.isnull().sum())

# Remove unnecessary columns
df.drop([
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Reason"
], axis=1, inplace=True)

print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Churn distribution
sns.countplot(x="Churn Label", data=df)

plt.title("Customer Churn Count")

plt.show()

#Contract vs Churn
sns.countplot(x="Contract", hue="Churn Label", data=df)

plt.xticks(rotation=20)

plt.show()

#Internet Service vs Churn
sns.countplot(x="Internet Service", hue="Churn Label", data=df)

plt.show()

#Monthly Charges Distribution
sns.histplot(df["Monthly Charges"], bins=30)

plt.show()

df["Churn Label"] = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn Label", axis=1)

# Save feature column names
joblib.dump(X.columns.tolist(), "model_columns.pkl")

y = df["Churn Label"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)


predictions = model.predict(X_test)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)


# Save model
joblib.dump(model, "churn_model.pkl")

print("Model saved successfully")

