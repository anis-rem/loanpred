import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("loan_data.csv")
print(df.head())

df['debt_to_income'] = df['loan_amnt'] / df['person_income']
df['income_per_age'] = df['person_income'] / (df['person_age'] + 1)

#data is mostly clean only thing needed to preprocess it is to categorize specific columns and make it suitable for the model
label_encoder = LabelEncoder()
for col in ['person_gender', 'person_education', 'person_home_ownership','loan_intent','previous_loan_defaults_on_file']:
    df[col] = label_encoder.fit_transform(df[col])

x=df.drop("loan_status", axis=1)
y=df["loan_status"]
print("Missing values:")
print(df.isnull().sum())
# pply feature weights
df['debt_to_income'] *= 2
df['person_income'] *= 1.5
df['previous_loan_defaults_on_file'] *= 2
df['loan_amnt'] *= 1.2
df['loan_intent'] *= 1.2
df['income_per_age'] *= 1.5
print(x.head())
print(y.head())
#splitting data into test and train
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
#initializing base models and training them on the training data
decsiontreemodel=DecisionTreeClassifier(random_state=42)
decsiontreemodel.fit(xtrain,ytrain)
preddeci=decsiontreemodel.predict(xtest)
randomforestmodel=RandomForestClassifier(random_state=42)
randomforestmodel.fit(xtrain,ytrain)
predrandfr=randomforestmodel.predict(xtest)
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(xtrain, ytrain)
#tuning up hyperparameters of randomforest using randomizedsearch for best result time and computational power wise
param_gridrf = {
    "n_estimators": [50, 100],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 4],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced", None]
}

rfrandomsearch = RandomizedSearchCV(estimator=randomforestmodel, param_distributions=param_gridrf,
                                      n_iter=30, cv=4, verbose=2, n_jobs=-1)
rfrandomsearch.fit(xtrain, ytrain)
rfmodelpred=rfrandomsearch.predict(xtest)
#same thing with decision tree
param_griddt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'max_features': [None, 'sqrt'],
    'max_leaf_nodes': [None, 20],
    'min_impurity_decrease': [0.0, 0.01]
}

dtrandomsearch = RandomizedSearchCV(estimator=decsiontreemodel, param_distributions=param_griddt,n_iter=30, cv=4, verbose=2, n_jobs=-1)
dtrandomsearch.fit(xtrain,ytrain)
dtmodelpred=dtrandomsearch.predict(xtest)
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2]
}

gb_randomsearch = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=param_grid_gb,
    n_iter=30,
    cv=4,
    verbose=2,
    n_jobs=-1
)
gb_randomsearch.fit(xtrain, ytrain)
gb_pred_tuned = gb_randomsearch.predict(xtest)
#printing out the results so we can compare between these two models
print("random forest results")
print("accuracy:", accuracy_score(ytest, rfmodelpred))
print("random forest report:\n", classification_report(ytest, rfmodelpred))
print("decision tree results")
print("accuracy:", accuracy_score(ytest, dtmodelpred))
print("decision tree report:\n", classification_report(ytest, dtmodelpred))
print("gradient boosting results")
print("accuracy:", accuracy_score(ytest, gb_pred_tuned))
print("gradient boosting lassification report:\n", classification_report(ytest, gb_pred_tuned))
base=LogisticRegression(random_state=42)
def compare_with_baseline(base, dt_model, xtrain, xtest, ytrain, ytest):
    base.fit(xtrain, ytrain)
    base_pred = base.predict(xtest)
    baseline_acc = accuracy_score(ytest,base_pred)
    print(f"baseline logistic regression accuracy: {baseline_acc:.4f}")

    dt_model.fit(xtrain, ytrain)
    dt_pred = dt_model.predict(xtest)
    dt_acc = accuracy_score(ytest, dt_pred)
    outcome = "outperformed" if dt_acc > baseline_acc else "underperformed"

    print(f"DecisionTreeClassifier Accuracy: {dt_acc:.4f} ({outcome})")
compare_with_baseline(base, dtrandomsearch, xtrain, xtest, ytrain, ytest)
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.grid(False)
    plt.show()
plot_confusion(ytest, rfmodelpred, "RF Confusion Matrix")
plot_confusion(ytest, dtmodelpred, "DT Confusion Matrix")
plot_confusion(ytest, gb_pred_tuned, "GB Confusion Matrix")
plt.figure(figsize=(12, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title(" Feature Correlation Heatmap")
plt.show()
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.title(" Loan Amount vs Loan Status")
plt.xlabel("Loan Status (0 = Denied, 1 = Approved)")
plt.ylabel("Loan Amount")
plt.show()
sns.boxplot(x='loan_status', y='person_income', data=df)
plt.title(" person_income vs Loan Status")
plt.xlabel("Loan Status (0 = Denied, 1 = Approved)")
plt.ylabel("Loan Amount")
plt.show()
joblib.dump(dtmodelpred, 'BESTLOANPRED.pkl')
joblib.dump(scaler, 'scaler.pkl')