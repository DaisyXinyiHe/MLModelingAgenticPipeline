# # Toy fallback manager that runs a minimal pipeline (EDA -> simple model -> eval -> report)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, root_mean_squared_error
import json
import numpy as np

class ManagerAgent:
    def __init__(self):
        self.name = 'Manager_Agent'

    def run_pipeline(self,user_instructions,user_defined_target,csv_path,varb_info_path):

        df = pd.read_csv(csv_path)
        # quick EDA
        eda = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "missing": df.isna().sum().to_dict()
        }

        ## Check if target is continous or categorical to determine regression or classification model
        if df[user_defined_target].nunique() > 10:
            model_type = "regression"
        else:
            model_type = "classificaiotn"

        # simple transform: drop rows with missing target, fill others with median/mode
        df = df.dropna(subset=[user_defined_target])
        for col in df.columns:
            if df[col].dtype.kind in "biufc":  # numeric-like
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(np.nan)

        # simple train/test split
        X = df.drop(columns=[user_defined_target])
        # keep only numeric columns for the toy example
        X_num = X.select_dtypes(include=["number"])
        if X_num.shape[1] == 0:
            raise RuntimeError("Toy pipeline needs at least one numeric predictor column.")
        y = df[user_defined_target]
        X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)

        metrics = {}

        if model_type =='regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics['rmse'] = root_mean_squared_error(y_test, y_pred)

        else:
            model = RandomForestClassifier(n_estimators=100,random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proab = model.predict_proba(X_test)
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred))
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))

        



        # produce simple report
        report = f"Toy pipeline report\nRows: {eda['n_rows']}\nCols: {eda['n_cols']}\nMetrics: {json.dumps(metrics)}\n"



        return {
            "eda": eda,
            "model_summary": {"type": "RandomForest", "details": "Toy model"},
            "evaluation": metrics,
            "report_markdown": report
        }


