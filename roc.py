import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier

# Load the dataset
data = pd.read_csv('Soil_and_Crop_Yield_Data_Nepal.csv')

# One-hot encoding for categorical data (if not already encoded)
data = pd.get_dummies(data)

# Assume 'Crop_Yield' is categorical for this example
y = data['Crop_Yield'].astype('category').cat.codes
X = data.drop('Crop_Yield', axis=1)
y = label_binarize(y, classes=np.unique(y))
n_classes = y.shape[1]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize OneVsRestClassifier with RandomForest and SVM
rf_clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
svm_clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))

# Train the classifiers
rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train_scaled, y_train)

# Predict probabilities
y_score_rf = rf_clf.predict_proba(X_test)
y_score_svm = svm_clf.predict_proba(X_test_scaled)

# Compute ROC curve and ROC area for each class for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test.ravel(), y_score_rf.ravel())
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Compute ROC curve and ROC area for each class for SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test.ravel(), y_score_svm.ravel())
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC Curves
plt.figure(figsize=(10, 5))
plt.plot(fpr_rf, tpr_rf, color='blue', label='Random Forest (area = %0.2f)' % roc_auc_rf)
plt.plot(fpr_svm, tpr_svm, color='red', label='SVM (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Compute Precision-Recall curve and AUC for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test.ravel(), y_score_rf.ravel())
ap_rf = average_precision_score(y_test, y_score_rf, average="micro")

# Compute Precision-Recall curve and AUC for SVM
precision_svm, recall_svm, _ = precision_recall_curve(y_test.ravel(), y_score_svm.ravel())
ap_svm = average_precision_score(y_test, y_score_svm, average="micro")

# Plot Precision-Recall curves
plt.figure(figsize=(10, 5))
plt.plot(recall_rf, precision_rf, color='blue', label='Random Forest AP=%0.2f' % ap_rf)
plt.plot(recall_svm, precision_svm, color='red', label='SVM AP=%0.2f' % ap_svm)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()
