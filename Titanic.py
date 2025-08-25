import pandas as pd
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt

# -------------------------------
# 1️ Load datasets
# -------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -------------------------------
# 2️ Preprocessing function
# -------------------------------
def preprocess(df, is_train=True):
    # Fill missing values (Data Imputation)
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Cabin → HasCabin
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df.drop('Cabin', axis=1, inplace=True)
    else:
        df['HasCabin'] = 0
    # Encode categorical columns
    for col in ['Sex','Embarked']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    # Drop unnecessary columns
    drop_cols = ['Name','Ticket']
    if is_train:
        drop_cols.append('PassengerId')
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df

# Apply preprocessing to both train and test
train_processed = preprocess(train)
test_processed = preprocess(test, is_train=False)

# -------------------------------
# 3️ Split features and target
# -------------------------------
x_train = train_processed.drop('Survived', axis=1)
y_train = train_processed['Survived']

# Ensure test has same columns
x_test = test_processed.reindex(columns=x_train.columns, fill_value=0)

# -------------------------------
# 4️ Train XGBoost model
# -------------------------------
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(x_train, y_train)

# -------------------------------
# 5️ Make predictions and save submission
# -------------------------------
predictions = model.predict(x_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")

# -------------------------------
# 6️ Visualize first tree (optional)
# -------------------------------
plt.figure(figsize=(30,20))
plot_tree(model, num_trees=0)  # Plot first tree
plt.show()
