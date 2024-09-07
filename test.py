import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('data/csv/data_milknew.csv')

# Display basic information
print(df.describe(include='all').to_string())
print(df.head())

# Encode target variable
label_encoder = LabelEncoder() 
df["Grade"] = label_encoder.fit_transform(df["Grade"])
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print('Grade [high = 0 / low = 1 / medium = 2] :')
print(mapping)
print(df.head().to_string())

# Split dataset
X = df.loc[:, df.columns != 'Grade']
y = df['Grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters grids
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs'], 'penalty': ['l2']}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
param_grid_svm = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}

# Instantiate models
logreg = LogisticRegression()
knn = KNeighborsClassifier()
svm = SVC()

# GridSearchCV for each model
grid_search_lr = GridSearchCV(estimator=logreg, param_grid=param_grid_lr, cv=5, n_jobs=-1, verbose=2, scoring='accuracy', refit='accuracy')
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, n_jobs=-1, verbose=2, scoring='accuracy', refit='accuracy')
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=2, scoring='accuracy', refit='accuracy')

# Fit models
grid_search_lr.fit(X_train_scaled, y_train)
grid_search_knn.fit(X_train_scaled, y_train)
grid_search_svm.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_lr = grid_search_lr.predict(X_test_scaled)
y_pred_knn = grid_search_knn.predict(X_test_scaled)
y_pred_svm = grid_search_svm.predict(X_test_scaled)

print("_____________________________________________________________", '\n')
print('Logistic Regression Best Parameters:', grid_search_lr.best_params_)
print('Logistic Regression Classification Report:\n', classification_report(y_test, y_pred_lr))
print("_____________________________________________________________", '\n')
print('KNN Best Parameters:', grid_search_knn.best_params_)
print('KNN Classification Report:\n', classification_report(y_test, y_pred_knn))
print("_____________________________________________________________", '\n')
print('SVM Best Parameters:', grid_search_svm.best_params_)
print('SVM Classification Report:\n', classification_report(y_test, y_pred_svm))
print("_____________________________________________________________", '\n')

# Save best models
models = {'logistic_regression': grid_search_lr, 'knn': grid_search_knn, 'svm': grid_search_svm}
for model_name, grid_search in models.items():
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'models/best_{model_name}.pkl')
    joblib.dump(mapping, 'models/mapping.pkl')
    joblib.dump(X.columns, 'models/columns.pkl')
    print(f"Saved best {model_name} model to 'models/best_{model_name}.pkl'")
