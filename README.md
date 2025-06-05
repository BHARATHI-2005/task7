Task 7: Support Vector Machines (SVM) - AI & ML Internship

Objective
Apply Support Vector Machines (SVM) for binary classification using both linear and non-linear (RBF) kernels, visualize decision boundaries (2D), and tune hyperparameters using cross-validation.

Dataset
Breast Cancer Wisconsin Dataset
- Target column: `diagnosis`  
  - Malignant (M) → 1  
  - Benign (B) → 0

Tools & Libraries Used
- Python
- Pandas, NumPy
- Scikit-learn (SVM, GridSearchCV, scaling, metrics)
- Matplotlib, Seaborn

Steps Followed

1. Data Preprocessing
- Loaded the dataset from CSV.
- Dropped irrelevant columns (e.g., ID).
- Encoded target values (M = 1, B = 0).
- Scaled features using `StandardScaler`.

2. Model Training
- Trained a **Linear SVM** using `SVC(kernel='linear')`.
- Trained an **RBF SVM** using `SVC(kernel='rbf')`.

3. Model Evaluation
- Evaluated both models using:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score

4. Hyperparameter Tuning
- Used `GridSearchCV` to find the best values of:
  - `C` (penalty parameter)
  - `gamma` (kernel coefficient for RBF)

5. Cross-Validation
- Performed 5-fold cross-validation to assess model stability.

6. Visualization
- Used only 2 features (`radius_mean`, `texture_mean`) to plot the SVM decision boundary in 2D space.

Results Summary

| Model        | Accuracy | Kernel | Cross-Validated |
|--------------|----------|--------|------------------|
| Linear SVM   | ~96%     | Linear | Yes              |
| RBF SVM      | ~98%     | RBF    | Yes              |
| GridSearchCV Best | ~99% | RBF + Tuned | Yes        |

Learning Outcomes
- Understood how margin maximization works.
- Practiced using kernel tricks.
- Tuned models to improve performance.
- Learned to visualize classification boundaries.

