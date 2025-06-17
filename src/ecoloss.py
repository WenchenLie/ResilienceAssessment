import numpy as np
from .building import Building
from .compenent import Component
from ._calculation import get_EDP_fragility
from config.config import EDP_TYPING, LOGGER


def intensity_based_loss(
    n: int,
    Sa_ls: np.ndarray,
    building: Building,
    is_random: bool = True
):
    for idx_Sa, Sa in enumerate(Sa_ls):
        for i in range(n):
            IDR: np.ndarray = building._simu_IDR(Sa)
            RIDR: float = building._simu_RIDR(Sa)
            PFA: np.ndarray = building._simu_PFA(Sa)
            PFV: np.ndarray = building._simu_PFV(Sa)
            clps = building._simu_clps(Sa)
            if clps:
                ...  # TODO: 倒塌
                continue
            dm = building._simu_demolishment(RIDR)
            if dm:
                ...  # TODO: 拆除
                continue
            # 可修复，进行损失计算
            for comp, quantity, story, floor in building.components:
                ds_ls: list[float] = comp.damage_states['median']
                beta_ls: list[float] = comp.damage_states['beta']
                edp_type: EDP_TYPING = comp.damage_states['edp_type']
                


def time_based_loss():
    ...

