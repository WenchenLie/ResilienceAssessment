import numpy as np
import pandas as pd
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
        for _ in range(n):
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
                edp_type: EDP_TYPING = comp.damage_states['edp_type']
                ds_flag = [i for i in range(1, len(median_ls) + 1)]  # 各个损伤状态的序号
                median_ls: list[float] = comp.damage_states['median']  # 各个损伤状态的edp中值
                beta_ls: list[float] = comp.damage_states['beta']  # 各个损伤状态edp的beta
                P_ds = []  # 各个损伤状态的概率
                for i in range(len(ds_flag)):
                    ...
                
                
                # ds_flag.insert(0, 0)  # 添加无损伤状态
                # median_ls.insert(0, 1)
                # beta_ls.insert(0, 0)
                # ds = zip(ds_flag, median_ls, beta_ls)  # 各个损伤状态的序号、median、beta
                # ds = sorted(ds, key=lambda x: x[1], reverse=True)  # 按照median降序排列
                # ds_flag, median_ls, beta_ls = zip(*ds)
                # P_ds = []  # 处于各个损伤状态的概率
                # for i in range(len(ds_flag)):
                #     if i != len(ds_flag) - 1:
                #         P_ds.append(median_ls[i] - median_ls[i + 1])
                #     else:
                #         P_ds.append(median_ls[-1])
                
                



def time_based_loss():
    ...

