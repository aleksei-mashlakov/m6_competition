"""
Encoders for encoding categorical variables and scaling continuous data.
"""
import warnings
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas.core.algorithms import isin
from sklearn.base import BaseEstimator, TransformerMixin
from torch.nn.utils import rnn


class NaNLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Labelencoder that can optionally always encode nan and unknown classes (in transform) as class ``0``
    """

    def __init__(self, add_nan: bool = False, warn: bool = True):
        """
        init OneHotLabelEncoder
        Args:
            add_nan: if to force encoding of nan at 0
            warn: if to warn if additional nans are added because items are unknown
        """
        self.add_nan = add_nan
        self.warn = warn
        super().__init__()

    def fit_transform(self, y: pd.Series, overwrite: bool = False) -> np.ndarray:
        """
        Fit and transform data.
        Args:
            y (pd.Series): input data
            overwrite (bool): if to overwrite current mappings or if to add to it.
        Returns:
            np.ndarray: encoded data
        """
        self.fit(y, overwrite=overwrite)
        return self.transform(y)

    @staticmethod
    def is_numeric(y: pd.Series) -> bool:
        """
        Determine if series is numeric or not. Will also return True if series is a categorical type with
        underlying integers.
        Args:
            y (pd.Series): series for which to carry out assessment
        Returns:
            bool: True if series is numeric
        """
        return y.dtype.kind in "bcif" or (
            isinstance(y, pd.CategoricalDtype) and y.cat.categories.dtype.kind in "bcif"
        )

    def fit(self, y: pd.Series, overwrite: bool = False):
        """
        Fit transformer
        Args:
            y (pd.Series): input data to fit on
            overwrite (bool): if to overwrite current mappings or if to add to it.
        Returns:
            OneHotLabelEncoder: self
        """
        #         if not overwrite and hasattr(self, "classes_"):
        #             offset = len(self.classes_)
        #         else:
        #             offset = 0
        #             self.classes_ = {}

        #         # determine new classes
        #         if self.add_nan:
        #             if self.is_numeric(y):
        #                 nan = np.nan
        #             else:
        #                 nan = "nan"
        #             self.classes_[nan] = 0
        #             idx = 1
        #         else:
        #             idx = 0

        #         idx += offset
        #         for val in np.unique(y):
        #             if val not in self.classes_:
        #                 self.classes_[val] = idx
        #                 idx += 1

        #         self.classes_vector_ = np.array(list(self.classes_.keys()))
        return self

    def transform(
        self,
        y: Iterable,
        return_norm: bool = False,
        target_scale=None,
        ignore_na: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode iterable with integers.
        Args:
            y (Iterable): iterable to encode
            return_norm: only exists for compatability with other encoders - returns a tuple if true.
            target_scale: only exists for compatability with other encoders - has no effect.
            ignore_na (bool): if to ignore na values and map them to zeros
                (this is different to `add_nan=True` option which maps ONLY NAs to zeros
                while this options maps the first class and NAs to zeros)
        Returns:
            Union[torch.Tensor, np.ndarray]: returns encoded data as torch tensor or numpy array depending on input type
        """
        if not isinstance(y, torch.Tensor):
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            elif isinstance(y, pd.Series):
                y = torch.from_numpy(y.values)

        encoded = F.one_hot(y)
        return encoded

    #         if self.add_nan:
    #             if self.warn:
    #                 cond = np.array([item not in self.classes_ for item in y])
    #                 if cond.any():
    #                     warnings.warn(
    #                         f"Found {np.unique(np.asarray(y)[cond]).size} unknown classes which were set to NaN",
    #                         UserWarning,
    #                     )

    #             encoded = [self.classes_.get(v, 0) for v in y]

    #         else:
    #             if ignore_na:
    #                 na_fill_value = next(iter(self.classes_.values()))
    #                 encoded = [self.classes_.get(v, na_fill_value) for v in y]
    #             else:
    #                 try:
    #                     encoded = [self.classes_[v] for v in y]
    #                 except KeyError as e:
    #                     raise KeyError(
    #                         f"Unknown category '{e.args[0]}' encountered. Set `add_nan=True` to allow unknown categories"
    #                     )

    #         if isinstance(y, torch.Tensor):
    #             encoded = torch.tensor(encoded, dtype=torch.long, device=y.device)
    #         else:
    #             encoded = np.array(encoded)

    #         if return_norm:
    #             return encoded, self.get_parameters()
    #         else:
    #             return encoded

    def inverse_transform(self, y: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Decode data, i.e. transform from integers to labels.
        Args:
            y (Union[torch.Tensor, np.ndarray]): encoded data
        Raises:
            KeyError: if unknown elements should be decoded
        Returns:
            np.ndarray: decoded data
        """
        #         if y.max() >= len(self.classes_vector_):
        #             raise KeyError("New unknown values detected")

        #         # decode
        #         decoded = self.classes_vector_[y]

        # decode
        decoded = torch.argmax(y, dim=1, keepdim=False)  # list, keepdim=True -> -1, 1

        return decoded.cpu().detach().numpy()

    def __call__(self, data: (Dict[str, torch.Tensor])) -> torch.Tensor:
        """
        Extract prediction from network output. Does not map back to input
        categories as this would require a numpy tensor without grad-abilities.
        Args:
            data (Dict[str, torch.Tensor]): Dictionary with entries
                * prediction: data to de-scale
        Returns:
            torch.Tensor: prediction
        """
        return data["prediction"]

    def get_parameters(self, groups=None, group_names=None) -> np.ndarray:
        """
        Get fitted scaling parameters for a given group.
        All parameters are unused - exists for compatability.
        Returns:
            np.ndarray: zero array.
        """
        return np.zeros(2, dtype=np.float64)


class OneHotTorchNormalizer(TorchNormalizer):
    """
    Basic target transformer that can be fit also on torch tensors.
    """

    # transformation and inverse transformation
    TRANSFORMATIONS = {
        "log": (torch.log, torch.exp),
        "log1p": (torch.log1p, torch.exp),
        "logit": (torch.logit, torch.sigmoid),
        "softplus": (_plus_one, F.softplus),
        "relu": (_identity, _clamp_zero),
        "onehot": (F.one_hot, torch.argmax),
    }

    def __init__(
        self,
        method: str = "standard",
        center: bool = True,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        eps: float = 1e-8,
    ):
        """
        Initialize
        Args:
            method (str, optional): method to rescale series. Either "identity", "standard" (standard scaling)
                or "robust" (scale using quantiles 0.25-0.75). Defaults to "standard".
            center (bool, optional): If to center the output to zero. Defaults to True.
            transformation (Union[str, Tuple[Callable, Callable]] optional): Transform values before
                applying normalizer. Available options are
                * None (default): No transformation of values
                * log: Estimate in log-space leading to a multiplicative model
                * logp1: Estimate in log-space but add 1 to values before transforming for stability
                    (e.g. if many small values <<1 are present).
                    Note, that inverse transform is still only `torch.exp()` and not `torch.expm1()`.
                * logit: Apply logit transformation on values that are between 0 and 1
                * softplus: Apply softplus to output (inverse transformation) and x + 1 to input (transformation)
                * relu: Apply max(0, x) to output
                * Tuple[Callable, Callable] of PyTorch functions that transforms and inversely transforms values.
            eps (float, optional): Number for numerical stability of calculations.
                Defaults to 1e-8.
        """
        super().__init__(
            method=method, center=center, transformation=transformation, eps=eps
        )


#     def get_parameters(self, *args, **kwargs) -> torch.Tensor:
#         """
#         Returns parameters that were used for encoding.
#         Returns:
#             torch.Tensor: First element is center of data and second is scale
#         """
#         return

#     def preprocess(
#         self, y: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]
#     ) -> Union[np.ndarray, torch.Tensor]:
#         """
#         Preprocess input data (e.g. take log).
#         Uses ``transform`` attribute to determine how to apply transform.
#         Returns:
#             Union[np.ndarray, torch.Tensor]: return rescaled series with type depending on input type
#         """
#         if self.transformation is None:
#             return y

#         if isinstance(y, torch.Tensor):
#             y = self.TRANSFORMATIONS.get(self.transformation, self.transformation)[0](y)
#         else:
#             # convert first to tensor, then transform and then convert to numpy array
#             if isinstance(y, (pd.Series, pd.DataFrame)):
#                 y = y.to_numpy()
#             y = torch.as_tensor(y)
#             y = self.TRANSFORMATIONS.get(self.transformation, self.transformation)[0](y)
#             y = np.asarray(y)
#         return y

#     def inverse_preprocess(self, y: Union[pd.Series, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
#         """
#         Inverse preprocess re-scaled data (e.g. take exp).
#         Uses ``transform`` attribute to determine how to apply inverse transform.
#         Returns:
#             Union[np.ndarray, torch.Tensor]: return rescaled series with type depending on input type
#         """
#         if self.transformation is None:
#             pass
#         elif isinstance(y, torch.Tensor):
#             y = self.TRANSFORMATIONS.get(self.transformation, self.transformation)[1](y)
#         else:
#             # convert first to tensor, then transform and then convert to numpy array
#             y = torch.as_tensor(y)
#             y = self.TRANSFORMATIONS.get(self.transformation, self.transformation)[1](y)
#             y = np.asarray(y)
#         return y

#     def fit(self, y: Union[pd.Series, np.ndarray, torch.Tensor]):
#         """
#         Fit transformer, i.e. determine center and scale of data
#         Args:
#             y (Union[pd.Series, np.ndarray, torch.Tensor]): input data
#         Returns:
#             TorchNormalizer: self
#         """
#         y = self.preprocess(y)
#         #self._set_parameters(y_center=y, y_scale=y)
#         return self

#     def _set_parameters(
#         self, y_center: Union[pd.Series, np.ndarray, torch.Tensor], y_scale: Union[pd.Series, np.ndarray, torch.Tensor]
#     ):
#         """
#         Calculate parameters for scale and center based on input timeseries
#         Args:
#             y_center (Union[pd.Series, np.ndarray, torch.Tensor]): timeseries for calculating center
#             y_scale (Union[pd.Series, np.ndarray, torch.Tensor]): timeseries for calculating scale
#         """
#         pass
# #         if self.method == "identity":
# #             if isinstance(y_center, torch.Tensor):
# #                 self.center_ = torch.zeros(y_center.size()[:-1])
# #                 self.scale_ = torch.ones(y_scale.size()[:-1])
# #             elif isinstance(y_center, (np.ndarray, pd.Series, pd.DataFrame)):
# #                 self.center_ = np.zeros(y_center.shape[:-1])
# #                 self.scale_ = np.ones(y_scale.shape[:-1])
# #             else:
# #                 self.center_ = 0.0
# #                 self.scale_ = 1.0

# #         elif self.method == "standard":
# #             if isinstance(y_center, torch.Tensor):
# #                 self.center_ = torch.mean(y_center, dim=-1)
# #                 self.scale_ = torch.std(y_scale, dim=-1) + self.eps
# #             elif isinstance(y_center, np.ndarray):
# #                 self.center_ = np.mean(y_center, axis=-1)
# #                 self.scale_ = np.std(y_scale, axis=-1) + self.eps
# #             else:
# #                 self.center_ = np.mean(y_center)
# #                 self.scale_ = np.std(y_scale) + self.eps
# #             # correct numpy scalar dtype promotion, e.g. fix type from `np.float32(0.0) + 1e-8` gives `np.float64(1e-8)`
# #             if isinstance(self.scale_, np.ndarray) and np.isscalar(self.scale_):
# #                 self.scale_ = self.scale_.astype(y_scale.dtype)

# #         elif self.method == "robust":
# #             if isinstance(y_center, torch.Tensor):
# #                 self.center_ = torch.median(y_center, dim=-1).values
# #                 q_75 = y_scale.kthvalue(int(len(y_scale) * 0.75), dim=-1).values
# #                 q_25 = y_scale.kthvalue(int(len(y_scale) * 0.25), dim=-1).values
# #             elif isinstance(y_center, np.ndarray):
# #                 self.center_ = np.median(y_center, axis=-1)
# #                 q_75 = np.percentile(y_scale, 75, axis=-1)
# #                 q_25 = np.percentile(y_scale, 25, axis=-1)
# #             else:
# #                 self.center_ = np.median(y_center)
# #                 q_75 = np.percentile(y_scale, 75)
# #                 q_25 = np.percentile(y_scale, 25)
# #             self.scale_ = (q_75 - q_25) / 2.0 + self.eps
# #         if not self.center:
# #             self.scale_ = self.center_
# #             if isinstance(y_center, torch.Tensor):
# #                 self.center_ = torch.zeros_like(self.center_)
# #             else:
# #                 self.center_ = np.zeros_like(self.center_)

# #         if (np.asarray(self.scale_) < 1e-7).any():
# #             warnings.warn(
# #                 "scale is below 1e-7 - consider not centering "
# #                 "the data or using data with higher variance for numerical stability",
# #                 UserWarning,
# #             )

#     def transform(
#         self,
#         y: Union[pd.Series, np.ndarray, torch.Tensor],
#         return_norm: bool = False,
#         target_scale: torch.Tensor = None,
#     ) -> Union[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray], Union[np.ndarray, torch.Tensor]]:
#         """
#         Rescale data.
#         Args:
#             y (Union[pd.Series, np.ndarray, torch.Tensor]): input data
#             return_norm (bool, optional): [description]. Defaults to False.
#             target_scale (torch.Tensor): target scale to use instead of fitted center and scale
#         Returns:
#             Union[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray], Union[np.ndarray, torch.Tensor]]: rescaled
#                 data with type depending on input type. returns second element if ``return_norm=True``
#         """
#         y = self.preprocess(y)
#         return y
#         # get center and scale
# #         if target_scale is None:
# #             target_scale = self.get_parameters().numpy()[None, :]
# #         center = target_scale[..., 0]
# #         scale = target_scale[..., 1]
# #         if y.ndim > center.ndim:  # multiple batches -> expand size
# #             center = center.view(*center.size(), *(1,) * (y.ndim - center.ndim))
# #             scale = scale.view(*scale.size(), *(1,) * (y.ndim - scale.ndim))

# #         # transform
# #         dtype = y.dtype
# #         y = (y - center) / scale
# #         try:
# #             y = y.astype(dtype)
# #         except AttributeError:  # torch.Tensor has `.type()` instead of `.astype()`
# #             y = y.type(dtype)

# #         # return with center and scale or without
# #         if return_norm:
# #             return y, target_scale
# #         else:
# #             return y

#     def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
#         """
#         Inverse scale.
#         Args:
#             y (torch.Tensor): scaled data
#         Returns:
#             torch.Tensor: de-scaled data
#         """
#         return self(dict(prediction=y, target_scale=self.get_parameters().unsqueeze(0)))

#     def __call__(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Inverse transformation but with network output as input.
#         Args:
#             data (Dict[str, torch.Tensor]): Dictionary with entries
#                 * prediction: data to de-scale
#                 * target_scale: center and scale of data
#         Returns:
#             torch.Tensor: de-scaled data
#         """
#         # ensure output dtype matches input dtype
#         dtype = data["prediction"].dtype

#         # inverse transformation with tensors
#         norm = data["target_scale"]

#         # use correct shape for norm
#         if data["prediction"].ndim > norm.ndim:
#             norm = norm.unsqueeze(-1)

#         # transform
#         y = data["prediction"] * norm[:, 1, None] + norm[:, 0, None]

#         y = self.inverse_preprocess(y)

#         # return correct shape
#         if data["prediction"].ndim == 1 and y.ndim > 1:
#             y = y.squeeze(0)
#         return y.type(dtype)
