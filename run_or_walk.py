from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("run_or_walk.csv")
data.head()
X = data.iloc[:, 5:]
Y = data["activity"]
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, random_state=10, test_size=0.30)
g_model = GaussianNB()
g_model.fit(train_x, train_y)

predicted_values = g_model.predict(test_x)
print("\nAccuracy Score\n")
print(metrics.accuracy_score(predicted_values, test_y))

print("\nClassification Score\n")
print(metrics.classification_report(predicted_values, test_y))
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gaussian Naive Bayes (All Features)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print AUC score
print(f"AUC Score for All Features: {roc_auc:.2f}")
# Predict the values for the test set
predicted_values = g_model.predict(test_x)

# Generate the confusion matrix
cm_all = confusion_matrix(test_y, predicted_values)

# Visualize the confusion matrix
disp_all = ConfusionMatrixDisplay(confusion_matrix=cm_all, display_labels=g_model.classes_)
disp_all.plot(cmap="Blues")
plt.title("Confusion Matrix - All Features")
plt.show()

# Print confusion matrix
print("Confusion Matrix - All Features")
print(cm_all)

# Repeat the model once using only the acceleration values and then using only the gyroscope

X_A = data.iloc[:, 5:8]
Y_A = data["activity"]
train_x_a, test_x_a, train_y_a, test_y_a = train_test_split(
    X_A, Y_A, random_state=10, test_size=0.30)
g_model.fit(train_x_a, train_y_a)
predicted_values_a = g_model.predict(test_x_a)
print("\nAccuracy Score\n")
print(metrics.accuracy_score(predicted_values_a, test_y_a))
print("\nClassification Score\n")
print(metrics.classification_report(predicted_values_a, test_y_a))
# Predict the probabilities for the positive class
predicted_proba_a = g_model.predict_proba(test_x_a)[:, 1]

# Compute the ROC curve and AUC
fpr_a, tpr_a, thresholds_a = roc_curve(test_y_a, predicted_proba_a, pos_label="running")
roc_auc_a = roc_auc_score(test_y_a, predicted_proba_a)
# Plot ROC curve for accelerationonly features
plt.figure(figsize=(8, 6))
plt.plot(fpr_a, tpr_a, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_a:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Acceleration Features')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print AUC score for accelerationonly model
print(f"AUC Score for Acceleration Features: {roc_auc_a:.2f}")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Generate the confusion matrix for acceleration-only model
cm_a = confusion_matrix(test_y_a, predicted_values_a)


# Visualize the confusion matrix
disp_a = ConfusionMatrixDisplay(confusion_matrix=cm_a, display_labels=g_model.classes_)
disp_a.plot(cmap="Blues")
plt.title("Confusion Matrix - Acceleration Features")
plt.show()

# Print confusion matrix for accelerationonly model
print("Confusion Matrix - Acceleration Features")
print(cm_a)
X_G = data.iloc[:, 8:]
Y_G = data["activity"]
train_x_g, test_x_g, train_y_g, test_y_g = train_test_split(
    X_G, Y_G, random_state=10, test_size=0.30)
g_model.fit(train_x_g, train_y_g)
predicted_values_g = g_model.predict(test_x_g)
print("\nAccuracy Score\n")
print(metrics.accuracy_score(predicted_values_g, test_y_g))
print("\nClassification Score\n")
print(metrics.classification_report(predicted_values_g, test_y_g))

# Train the model with gyroscopeonly data
g_model.fit(train_x_g, train_y_g)

# Predict the probabilities for the positive class
predicted_proba_g = g_model.predict_proba(test_x_g)[:, 1]

# Compute the ROC curve and AUC
fpr_g, tpr_g, thresholds_g = roc_curve(test_y_g, predicted_proba_g, pos_label="running")
roc_auc_g = roc_auc_score(test_y_g, predicted_proba_g)
# Plot ROC curve for gyroscope-only features
plt.figure(figsize=(8, 6))
plt.plot(fpr_g, tpr_g, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_g:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gyroscope Features')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print AUC score for gyroscopeonly
print(f"AUC Score for Gyroscope Features: {roc_auc_g:.2f}")
#the confusion matrix for gyroscopeonly model
cm_g = confusion_matrix(test_y_g, predicted_values_g)


disp_g = ConfusionMatrixDisplay(confusion_matrix=cm_g, display_labels=g_model.classes_)
disp_g.plot(cmap="Blues")
plt.title("Confusion Matrix - Gyroscope Features")
plt.show()

# Print confusion matrix for gyroscopeonly model
print("Confusion Matrix - Gyroscope Features")
print(cm_g)

print("\nAccuracy Score (All Features):", metrics.accuracy_score(predicted_values, test_y))
print("\nAccuracy Score (Acceleration Features):", metrics.accuracy_score(predicted_values_a, test_y_a))
print("\nAccuracy Score (Gyroscope Features):", metrics.accuracy_score(predicted_values_g, test_y_g))
