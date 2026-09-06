from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax

class TemperatureScaler:
    def __init__(self, multiclass: bool):
        self.multiclass = bool(multiclass)
        self.temperature_ = 1.0
    def fit(self, probs, y):
        p = np.clip(np.asarray(probs, float), 1e-12, 1-1e-12)
        y = np.asarray(y)
        if self.multiclass:
            logits = np.log(p)
            def loss(log_t):
                q = softmax(logits / np.exp(log_t), axis=1)
                return float(-np.mean(np.log(q[np.arange(len(y)), y] + 1e-15)))
        else:
            p = p.reshape(-1)
            logits = np.log(p/(1-p))
            yb = y.astype(float).reshape(-1)
            def loss(log_t):
                q = expit(logits / np.exp(log_t))
                return float(-np.mean(yb*np.log(q+1e-15)+(1-yb)*np.log(1-q+1e-15)))
        r = minimize_scalar(loss, bounds=(-4,4), method='bounded')
        self.temperature_ = float(np.exp(r.x))
        return self
    def transform(self, probs):
        p = np.clip(np.asarray(probs, float), 1e-12, 1-1e-12)
        if self.multiclass:
            return softmax(np.log(p)/self.temperature_, axis=1)
        p = p.reshape(-1)
        return expit(np.log(p/(1-p))/self.temperature_)

@dataclass
class TabularModelBundle:
    preprocessor: Any
    model: Any
    task: str
    output_columns: List[str]
    model_kind: str
    class_labels: Optional[List[Any]] = None
    calibrator: Optional[TemperatureScaler] = None
    calibrated: bool = False
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self.preprocessor.transform(frame)
        if self.model_kind == 'multiclass':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message=r'X does not have valid feature names, but LGBMClassifier was fitted with feature names')
                p = np.asarray(self.model.predict_proba(x), float)
            if self.class_labels is not None:
                cur = list(self.model.classes_)
                p = p[:, [cur.index(v) for v in self.class_labels]]
            if self.calibrated and self.calibrator is not None:
                p = self.calibrator.transform(p)
            return pd.DataFrame(p, columns=self.output_columns, index=frame.index)
        if self.model_kind == 'binary':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message=r'X does not have valid feature names, but LGBMClassifier was fitted with feature names')
                p = np.asarray(self.model.predict_proba(x), float)
            pos = list(self.model.classes_).index(1)
            p1 = p[:, pos]
            if self.calibrated and self.calibrator is not None:
                p1 = self.calibrator.transform(p1)
            if len(self.output_columns) == 2:
                return pd.DataFrame({self.output_columns[0]:p1,self.output_columns[1]:1-p1}, index=frame.index)
            return pd.DataFrame({self.output_columns[0]:p1}, index=frame.index)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'X does not have valid feature names, but LGBMRegressor was fitted with feature names')
            pred = np.asarray(self.model.predict(x), float).reshape(-1)
        return pd.DataFrame({self.output_columns[0]:pred}, index=frame.index)

@dataclass
class GoalBundle:
    home: TabularModelBundle
    away: TabularModelBundle
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([self.home.predict(frame), self.away.predict(frame)], axis=1)
