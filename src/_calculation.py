from math import isclose, erf, sqrt
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.config import LOGGER


def get_EDP_fragility(
    Sa_ls: np.ndarray,
    edp_ls: np.ndarray,
    gm_num: int,
    IDA_data: list[np.ndarray]
):
    if isclose(edp_ls[0], 0):
        raise ValueError('EDP值不能以0开头')
    if isclose(Sa_ls[0], 0):
        raise ValueError('Sa值不能以0开头')
    for idx_edp, edp in enumerate(edp_ls):
        IM_ls = np.zeros(gm_num)  # 给定EDP值下的IM值
        for idx_gm, curve in enumerate(IDA_data):
            x, y = curve[:, 0], curve[:, 1]  # 单条IDA曲线的横、纵坐标
            if not min(x) <= edp <= max(x):
                raise ValueError(f'EDP值({edp})超出IDA曲线范围({min(x)}, {max(x)})')
            yi = _get_y(x, y, edp)  # 给定EDP值下，IDA曲线对应的地震强度
            IM_ls[idx_gm] = yi


def _get_y(x: list, y: list, x0: float, error: bool=True) -> float:
    """获得竖线x=x0与给定曲线的交点纵坐标

    Args:
        x (list): 输入曲线的横坐标序列
        y (list): 输入曲线的纵坐标序列
        x0 (float): 竖直线x = x0
        error (boo, optional): 若x0超出范围，抛出异常or返回None

    Returns:
        float: 曲线与竖线交点纵坐标
    """
    # 获得x=x0与曲线的交点
    if x0 < min(x):
        if error:
            raise ValueError(f'【Error】x0 < min(x) ({x0} < {min(x)})')
        else:
            return None
    if x0 > max(x):
        if error:
            raise ValueError(f'【Error】x0 > max(x) ({x0} > {max(x)})')
        else:
            return None
    for i in range(len(x) - 1):
        if x[i] == x0:
            y0 = y[i]
            return y0
        elif x[i] < x0 <= x[i + 1]:
            k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            y0 = k * (x0 - x[i]) + y[i]
            return y0
    else:
        raise ValueError('【Error】未找到交点-2')


def _cdf(x: float):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal(mean: float, std: float, trunc=False) -> float:
    """正态分布采样

    Args:
        mean (float): 均值
        std (float): 标准差
        trunc (bool, optional): 是否按10%-90%分位数进行截断采样

    Returns:
        float: 采样值
    """
    x = np.random.normal(mean, std)
    if trunc:
        if not mean - 1.28155 * std < x < mean + 1.28155 * std:
            x = _normal(mean, std, trunc)
    return float(x)


def _lognormal(mean: float, log_std: float, trunc=False) -> float:
    """正态分布采样

    Args:
        mean (float): 均值
        log_std (float): 对数标准差
        trunc (bool, optional): 是否按10%-90%分位数进行截断采样

    Returns:
        float: 采样值
    """
    log_mean = np.log(mean)
    x = np.random.lognormal(log_mean, log_std)
    if trunc:
        if not log_mean - 1.28155 * log_std < np.log(x) < log_mean + 1.28155 * log_std:
            x = _lognormal(mean, log_std, trunc)
    return float(x)