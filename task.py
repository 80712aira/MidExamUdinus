import pandas as pd

# Load dataset
df = pd.read_csv('nama_file_dataset.csv')
# Masukkan nama fitur ke dalam dataset
df.columns = ['nama_fitur_1', 'nama_fitur_2', ... , 'income']
# Cek dan rubah data yang kosong menjadi NaN
df.replace('?', np.nan, inplace=True)
# Drop data duplikat
df.drop_duplicates(inplace=True)
# Drop fitur yang bernilai konstan
df = df.loc[:,df.apply(pd.Series.nunique) != 1]
# Visualisasi 22 fitur dengan barchart
import matplotlib.pyplot as plt

df.iloc[:, :22].value_counts().plot(kind='bar')
plt.show()
# Cek fitur yang bertipe kategorikal
categorical_features = df.select_dtypes(include=['object']).columns
# Konversikan nilai kategorikal menjadi ordinal
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    df[feature] = label_encoders[feature].fit_transform(df[feature].astype(str))
# Cek korelasi
correlation_matrix = df.corr()
from sklearn.model_selection import train_test_split

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 7, 9, 11]}
model = DecisionTreeClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Prediksi menggunakan model terbaik
best_model = DecisionTreeClassifier(max_depth=best_params['max_depth'])
best_model.fit(X_train, y_train)
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluasi performa model
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)

accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
