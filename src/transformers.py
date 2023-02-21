import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer

# https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Extract day of year and time of day features from a timestamp
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Timestamps must be in index
        X = (
            X
            # .assign(hour = X.index.hour)
            .assign(day=X.index.day)
            .assign(day_of_week=X.index.dayofweek)
            .assign(week_of_year=X.index.week)
            .assign(month=X.index.month)
            #              .assign(is_month_end = X.index.is_month_end)
            #              .assign(is_month_start = X.index.is_month_start)
            #              .assign(is_quarter_end = X.index.is_quarter_end)
            #              .assign(is_quarter_start = X.index.is_quarter_start)
            .assign(dayofyear=X.index.dayofyear)
            #              .assign(is_year_start = X.index.is_year_start)
            #              .assign(is_year_end = X.index.is_year_end)
        )
        return X


# https://goldinlocks.github.io/Time-Series-Cross-Validation/
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]