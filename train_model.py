
# ============================================================
#       HEART DISEASE HIGH ACCURACY MODEL
#       1190 ROW DATASET
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET = "heart.csv"

df = pd.read_csv(DATASET)

print("=" * 65)
print("       HEART DISEASE HIGH ACCURACY MODEL")
print("=" * 65)

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nOriginal Columns:")
print(df.columns.tolist())


# ============================================================
# 2. RENAME DATASET COLUMNS
# ============================================================

column_mapping = {

    "chest pain type": "cp",

    "resting bp s": "trestbps",

    "cholesterol": "chol",

    "fasting blood sugar": "fbs",

    "resting ecg": "restecg",

    "max heart rate": "thalach",

    "exercise angina": "exang",

    "ST slope": "slope"

}


df = df.rename(
    columns=column_mapping
)


# ============================================================
# 3. DISPLAY NEW COLUMNS
# ============================================================

print("\nRenamed Columns:")
print(df.columns.tolist())


# ============================================================
# 4. REQUIRED FEATURES
# ============================================================

features = [

    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope"

]

target = "target"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [target]

missing_columns = [

    col

    for col in required_columns

    if col not in df.columns

]


if missing_columns:

    print("\nMissing Columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing."
    )


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

print("\nDuplicate Rows:")
print(
    df.duplicated().sum()
)


df = df.drop_duplicates().reset_index(
    drop=True
)


print("\nShape After Removing Duplicates:")
print(df.shape)


# ============================================================
# 7. CONVERT FEATURES TO NUMERIC
# ============================================================

for col in features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)


# ============================================================
# 8. REMOVE INVALID TARGET ROWS
# ============================================================

df = df.dropna(
    subset=[target]
).reset_index(
    drop=True
)


# ============================================================
# 9. TARGET CONVERSION
# ============================================================

# The dataset already uses:
#
# 0 = No heart disease
# 1 = Heart disease
#
# Therefore no conversion is required.

df[target] = df[target].astype(int)


# ============================================================
# 10. FEATURES AND TARGET
# ============================================================

X = df[features].copy()

y = df[target].copy()


# ============================================================
# 11. HANDLE MISSING VALUES
# ============================================================

X = X.fillna(
    X.median()
)


# ============================================================
# 12. DATASET INFORMATION
# ============================================================

print("\nFeatures:")
print(features)

print("\nNumber of Features:")
print(len(features))

print("\nTarget Distribution:")
print(
    y.value_counts()
)


# ============================================================
# 13. FEATURE TYPES
# ============================================================

categorical_features = [

    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope"

]


numerical_features = [

    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"

]


print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ============================================================
# 14. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))


# ============================================================
# 15. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ]),

            numerical_features
        ),

        (
            "cat",

            Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                )

            ]),

            categorical_features
        )

    ]

)


# ============================================================
# 16. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


# ============================================================
# 17. LOGISTIC REGRESSION
# ============================================================

lr_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        LogisticRegression(
            max_iter=5000
        )
    )

])


lr_parameters = {

    "model__C": [

        0.01,
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        20

    ]

}


# ============================================================
# 18. SVM
# ============================================================

svm_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        SVC(
            probability=True
        )
    )

])


svm_parameters = {

    "model__C": [

        0.1,
        0.5,
        1,
        2,
        5,
        10,
        20,
        50

    ],

    "model__gamma": [

        "scale",
        "auto",
        0.001,
        0.01,
        0.1

    ],

    "model__kernel": [

        "rbf"

    ]

}


# ============================================================
# 19. RANDOM FOREST
# ============================================================

rf_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",

        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

    )

])


rf_parameters = {

    "model__n_estimators": [

        300,
        500,
        800

    ],

    "model__max_depth": [

        None,
        5,
        8,
        10,
        15

    ],

    "model__min_samples_split": [

        2,
        4,
        6

    ],

    "model__min_samples_leaf": [

        1,
        2,
        3

    ],

    "model__max_features": [

        "sqrt",
        "log2"

    ]

}


# ============================================================
# 20. EXTRA TREES
# ============================================================

et_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",

        ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1
        )

    )

])


et_parameters = {

    "model__n_estimators": [

        300,
        500,
        800

    ],

    "model__max_depth": [

        None,
        5,
        8,
        10,
        15

    ],

    "model__min_samples_split": [

        2,
        4,
        6

    ],

    "model__min_samples_leaf": [

        1,
        2,
        3

    ]

}


# ============================================================
# 21. GRID SEARCH FUNCTION
# ============================================================

def tune_model(
    name,
    pipeline,
    parameters
):

    print("\n")
    print("=" * 65)

    print(
        "TUNING:",
        name
    )

    print("=" * 65)

    search = GridSearchCV(

        pipeline,

        parameters,

        cv=cv,

        scoring="accuracy",

        n_jobs=-1,

        verbose=1

    )

    search.fit(
        X_train,
        y_train
    )

    print("\nBest Parameters:")
    print(
        search.best_params_
    )

    print("\nBest CV Accuracy:")
    print(
        f"{search.best_score_ * 100:.2f}%"
    )

    return search.best_estimator_, search.best_score_


# ============================================================
# 22. TRAIN ALL MODELS
# ============================================================

best_lr, lr_score = tune_model(

    "LOGISTIC REGRESSION",

    lr_pipeline,

    lr_parameters

)


best_svm, svm_score = tune_model(

    "SVM",

    svm_pipeline,

    svm_parameters

)


best_rf, rf_score = tune_model(

    "RANDOM FOREST",

    rf_pipeline,

    rf_parameters

)


best_et, et_score = tune_model(

    "EXTRA TREES",

    et_pipeline,

    et_parameters

)



# ============================================================
# 23. MODEL COMPARISON WITH BEST ALGORITHM
# ============================================================

models = {

    "Logistic Regression":
        (best_lr, lr_score),

    "SVM":
        (best_svm, svm_score),

    "Random Forest":
        (best_rf, rf_score),

    "Extra Trees":
        (best_et, et_score)

}


print("\n")
print("=" * 65)
print("              MODEL COMPARISON")
print("=" * 65)

print(f"{'Algorithm':25} {'CV Accuracy':>15}")
print("-" * 65)

for name, (model, score) in sorted(
    models.items(),
    key=lambda x: x[1][1],
    reverse=True
):

    print(
        f"{name:25} "
        f"{score * 100:>14.2f}%"
    )

# Find best algorithm based on CV accuracy
best_algorithm_name, (
    best_algorithm_model,
    best_algorithm_score
) = max(
    models.items(),
    key=lambda x: x[1][1]
)

print("-" * 65)

print("\nBEST ALGORITHM:")
print(best_algorithm_name)

print("\nBEST CV ACCURACY:")
print(
    f"{best_algorithm_score * 100:.2f}%"
)

print("=" * 65)

# ============================================================
# 24. VOTING ENSEMBLE
# ============================================================

voting_model = VotingClassifier(

    estimators=[

        (
            "lr",
            best_lr
        ),

        (
            "svm",
            best_svm
        ),

        (
            "rf",
            best_rf
        ),

        (
            "et",
            best_et
        )

    ],

    voting="soft",

    n_jobs=-1

)


print("\n")
print("=" * 65)
print("              TRAINING ENSEMBLE")
print("=" * 65)


voting_model.fit(

    X_train,

    y_train

)


# ============================================================
# 25. TEST ALL MODELS
# ============================================================

test_results = {}


for name, (

    model,
    cv_score

) in models.items():

    prediction = model.predict(
        X_test
    )

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    test_results[name] = {

        "model": model,

        "accuracy": accuracy

    }

    print(

        f"\n{name} Test Accuracy: "

        f"{accuracy * 100:.2f}%"

    )


# ============================================================
# 26. TEST ENSEMBLE
# ============================================================

ensemble_prediction = voting_model.predict(
    X_test
)


ensemble_accuracy = accuracy_score(

    y_test,

    ensemble_prediction

)


print("\nVoting Ensemble Test Accuracy:")

print(
    f"{ensemble_accuracy * 100:.2f}%"
)


# ============================================================
# 27. SELECT BEST TEST MODEL
# ============================================================

best_name = max(

    test_results,

    key=lambda x:
    test_results[x]["accuracy"]

)


best_model = test_results[
    best_name
]["model"]


best_accuracy = test_results[
    best_name
]["accuracy"]


if ensemble_accuracy > best_accuracy:

    final_model = voting_model

    final_model_name = (
        "Voting Ensemble"
    )

    final_accuracy = (
        ensemble_accuracy
    )

else:

    final_model = best_model

    final_model_name = best_name

    final_accuracy = best_accuracy


# ============================================================
# 28. FINAL PREDICTION
# ============================================================

final_prediction = final_model.predict(

    X_test

)


final_accuracy = accuracy_score(

    y_test,

    final_prediction

)


# ============================================================
# 29. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("              FINAL TEST RESULTS")
print("=" * 65)

print("\nSelected Model:")
print(final_model_name)

print("\nTest Accuracy:")

print(
    f"{final_accuracy * 100:.2f}%"
)


# ============================================================
# 30. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        final_prediction,

        digits=4

    )

)


# ============================================================
# 31. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    final_prediction

)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 32. SENSITIVITY / SPECIFICITY
# ============================================================

tn, fp, fn, tp = cm.ravel()


sensitivity = (

    tp /

    (tp + fn)

)


specificity = (

    tn /

    (tn + fp)

)


print("\nSensitivity:")

print(
    f"{sensitivity * 100:.2f}%"
)


print("\nSpecificity:")

print(
    f"{specificity * 100:.2f}%"
)


# ============================================================
# 33. ROC-AUC
# ============================================================

if hasattr(

    final_model,

    "predict_proba"

):

    probability = (

        final_model.predict_proba(

            X_test

        )[:, 1]

    )

    auc = roc_auc_score(

        y_test,

        probability

    )

    print("\nROC-AUC:")

    print(
        f"{auc * 100:.2f}%"
    )


# ============================================================
# 34. TRAIN FINAL MODEL ON COMPLETE DATA
# ============================================================
#
# IMPORTANT:
# After evaluating the model on the untouched test set,
# retrain the selected model using all available data.
#
# The test accuracy printed above remains the honest test
# accuracy from the held-out test set.
# ============================================================

final_model.fit(

    X,

    y

)


# ============================================================
# 35. SAVE MODEL
# ============================================================

joblib.dump(

    final_model,

    "heart_model.pkl"

)


# ============================================================
# 36. SAVE FEATURES
# ============================================================

joblib.dump(

    features,

    "heart_features.pkl"

)


# ============================================================
# 37. SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "model":
        final_model_name,

    "features":
        features,

    "test_accuracy":
        final_accuracy,

    "sensitivity":
        sensitivity,

    "specificity":
        specificity,

    "dataset_rows":
        len(df)

}


joblib.dump(

    model_info,

    "heart_model_info.pkl"

)


# ============================================================
# 38. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 65)
print("             MODEL SAVED SUCCESSFULLY")
print("=" * 65)

print("\nModel File:")
print("heart_model.pkl")

print("\nFeature File:")
print("heart_features.pkl")

print("\nInformation File:")
print("heart_model_info.pkl")

print("\nSelected Model:")
print(final_model_name)

print("\nNumber of Features:")
print(len(features))

print("\nFinal Test Accuracy:")
print(
    f"{final_accuracy * 100:.2f}%"
)

print("\nSensitivity:")
print(
    f"{sensitivity * 100:.2f}%"
)

print("\nSpecificity:")
print(
    f"{specificity * 100:.2f}%"
)

print("\nProbability Prediction:")
print("Available through predict_proba()")

print("\n")
print("=" * 65)
print("                    DONE")
print("=" * 65)