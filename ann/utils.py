from typing import Union

import numpy as np
import pandas as pd


array = Union[np.ndarray, pd.DataFrame, pd.Series, list, tuple, dict]


def convert_to_array(x: array) -> np.ndarray:
    """
    Преобразование объекта в массив

    :param x: Объект
    :return: Массив
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=np.longdouble)
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.longdouble)
    if isinstance(x, list):
        return np.array(x, dtype=np.longdouble)
    if isinstance(x, tuple):
        return np.array(x, dtype=np.longdouble)
    if isinstance(x, dict):
        return np.array(list(x.values()), dtype=np.longdouble)
    if isinstance(x, int):
        return np.array([x], dtype=np.longdouble)
    if isinstance(x, float):
        return np.array([x], dtype=np.longdouble)
    raise ValueError(f"Не удалось преобразовать объект {x} типа {type(x)} в массив")
