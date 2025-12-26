import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # <-- new import

# ================================
# 1) SIMPLE STUDENT DATASET
# ================================
# Columns: ID, StudyHours, AttendancePercent, PreviousScore, ExtraClasses, PassStatus
data = np.array([
    [1,2,60,45,0,0],
    [2,5,70,55,0,0],
    [3,1,50,30,0,0],
    [4,7,80,65,1,1],
    [5,3,65,48,0,0],
    [6,9,90,78,1,1],
    [7,4,75,58,0,1],
    [8,6,85,62,1,1],
    [9,2,55,40,0,0],
    [10,8,88,70,1,1],
    [11,1,40,25,0,0],
    [12,10,95,82,1,1],
    [13,3,60,50,0,0],
    [14,4,78,55,1,1],
    [15,6,82,68,1,1],
    [16,2,58,36,0,0],
    [17,7,88,72,1,1],
    [18,5,74,60,0,1],
    [19,8,92,80,1,1],
    [20,3,68,52,0,0]
])

# ================================
# 2) EXTRACT FEATURES AND LABEL
# ================================
student_ids = data[:,0]
study_hours = data[:,1]
attendance = data[:,2]
previous_score = data[:,3]
extra_classes = data[:,4]
pass_status = data[:,5]

# Feature matrix
features = np.column_stack((study_hours, attendance, previous_score, extra_classes))

# ================================
# 3) NORMALIZE FEATURES
# ================================
mean_features = features.mean(axis=0)
std_features = features.std(axis=0)
features_normalized = (features - mean_features)/std_features

# ================================
# 4) LOGISTIC REGRESSION SETUP
# ================================
weights = np.zeros(features.shape[1])
bias = 0
learning_rate = 0.1
epochs = 2000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X):
    return sigmoid(np.dot(X, weights) + bias)

# ================================
# 5) TRAIN THE MODEL
# ================================
for _ in range(epochs):
    predicted = predict(features_normalized)
    gradient_weights = np.dot(features_normalized.T, (predicted - pass_status)) / len(pass_status)
    gradient_bias = np.mean(predicted - pass_status)
    weights -= learning_rate * gradient_weights
    bias -= learning_rate * gradient_bias

# ================================
# 6) MODEL PARAMETERS & FORMULAS
# ================================
print("=== Logistic Regression Model Parameters ===")
for i, w in enumerate(weights):
    print(f"Weight for feature {i+1}: {round(w,4)}")
print(f"Bias: {round(bias,4)}\n")

formula_pass = f"P(Pass) = 1 / (1 + exp(-({round(weights[0],4)}*StudyHours + {round(weights[1],4)}*Attendance + {round(weights[2],4)}*PreviousScore + {round(weights[3],4)}*ExtraClasses + {round(bias,4)})))"
formula_fail = f"P(Fail) = 1 - P(Pass) = 1 - ({formula_pass[8:]})"
print("=== Logistic Regression Formulas ===")
print(formula_pass)
print(formula_fail)

# ================================
# 7) MODEL ACCURACY
# ================================
predicted_labels = (predict(features_normalized) >= 0.5).astype(int)
accuracy = np.mean(predicted_labels == pass_status)
print("\nModel Accuracy on dataset:", round(accuracy*100,2), "%")

# ================================
# 8) PREDICT NEW STUDENT
# ================================
print("\nEnter new student details to predict Pass/Fail:")
new_hours = float(input("Study Hours (0-12): "))
new_attendance = float(input("Attendance % (20-100): "))
new_prev_score = float(input("Previous Exam Score (0-100): "))
new_extra = int(input("Extra Classes? (0=No, 1=Yes): "))

new_student = np.array([new_hours, new_attendance, new_prev_score, new_extra])
new_student_normalized = (new_student - mean_features)/std_features
predicted_prob = predict(new_student_normalized)
new_student_result = 1 if predicted_prob >= 0.5 else 0

print("\nPredicted Probability of Pass:", round(predicted_prob,2))
print("Predicted Result:", "PASS" if predicted_prob >= 0.5 else "FAIL")

# ================================
# 9) GRAPHS FOR DATASET
# ================================

# 9a) Pass/Fail Count Bar Graph
fail_count = np.sum(pass_status == 0)
pass_count = np.sum(pass_status == 1)
plt.figure(figsize=(5,4))
bars = plt.bar(["Fail","Pass"], [fail_count, pass_count], color=["red","green"])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')
plt.title("Number of Students: Pass vs Fail (Dataset)")
plt.ylabel("Number of Students")
plt.show()

# 9b) Study Hours vs Previous Score Scatter
colors = ['red' if x==0 else 'green' for x in pass_status]
plt.figure(figsize=(5,4))
plt.scatter(study_hours, previous_score, c=colors, alpha=0.7)
plt.xlabel("Study Hours")
plt.ylabel("Previous Exam Score")
plt.title("Pass/Fail Scatter: Study Hours vs Previous Score")
plt.grid(True)
plt.show()

# 9c) Attendance vs Predicted Pass Probability
predicted_all = predict(features_normalized)
plt.figure(figsize=(5,4))
plt.scatter(attendance, predicted_all, color="purple", alpha=0.7)
plt.xlabel("Attendance %")
plt.ylabel("Predicted Pass Probability")
plt.title("Attendance vs Predicted Pass Probability")
plt.grid(True)
plt.show()

# 9d) Combined Graph: Previous Score + Attendance
plt.figure(figsize=(6,5))
plt.scatter(previous_score, attendance, c=colors, alpha=0.7)
plt.xlabel("Previous Exam Score")
plt.ylabel("Attendance %")
plt.title("Combined Graph: Previous Score + Attendance")
plt.grid(True)
plt.show()

# 9e) Predicted Result Bar Graph for New Student
fail_bar = 1 if new_student_result == 0 else 0
pass_bar = 1 if new_student_result == 1 else 0
plt.figure(figsize=(5,4))
bars = plt.bar(["Fail","Pass"], [fail_bar, pass_bar], color=["red","green"])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')
plt.title("Predicted Result for New Student")
plt.ylabel("Count")
plt.show()

# ================================
# 10) CONFUSION MATRIX
# ================================
cm = confusion_matrix(pass_status, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail","Pass"])
plt.figure(figsize=(5,4))
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title("Confusion Matrix: Logistic Regression Predictions")
plt.show()