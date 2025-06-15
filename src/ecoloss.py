import numpy as np
from .building import Building
from .compenent import Component
from ._calculation import get_EDP_fragility


def intensity_based_loss(
    Sa_ls: np.ndarray,
    building: Building,
):
    ...


def time_based_loss():
    ...

