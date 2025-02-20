import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score

# Load dataset
data = pd.read_csv("train_features_extended.csv")  # Replace with actual feature file

# Drop non-numeric columns
data_numeric = data.select_dtypes(include=[np.number])

# Feature Selection: Remove redundant features
corr_matrix = data_numeric.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
data_numeric = data_numeric.drop(columns=to_drop)
print(f"Dropped highly correlated features: {to_drop}")

# Normalization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Clustering: K-Means & DBSCAN
kmeans = KMeans(n_clusters=3, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data_scaled)
print(f"K-Means Silhouette Score: {silhouette_score(data_scaled, data['KMeans_Cluster'])}")

dbscan = DBSCAN(eps=1.5, min_samples=5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(data_scaled)

# Visualization of clusters
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data['KMeans_Cluster'])
plt.title("K-Means Clustering")
plt.show()

# Classification: SVM & Random Forest
if 'Label' in data.columns:
    X_train, X_test, y_train, y_test = train_test_split(
        data_numeric, data['Label'], test_size=0.2, random_state=42
    )

    # Support Vector Machine
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save processed data
data.to_csv("processed_features.csv", index=False)
