from typing import Literal
from config.config import UNITS_TYPING


def to_foot(
        value: float | tuple[float] | list[float],
        old_unit: UNITS_TYPING
    ) -> float | tuple[float] | list[float]:
    """将数字、元组或列表中的值从旧单位转换为英尺(ft)

    Args:
        value (float | tuple[float] | list[float]): 数值
        old_unit (UNITS_TYPING): 原单位

    Returns:
        float | tuple[float] | list[float]: 新数值
    """
    sf = 1  # 单位换算系数
    if old_unit == 'ft':
        sf = 1
    elif old_unit == 'in':
        sf = 12
    elif old_unit =='mm':
        sf = 12 / 0.3048
    elif old_unit =='m':
        sf = 12 / 0.3048 / 1000
    else:
        raise ValueError(f"Invalid unit: {old_unit}")
    if isinstance(value, tuple):
        return tuple(v * sf for v in value)
    elif isinstance(value, list):
        return [v * sf for v in value]
    elif isinstance(value, (float, int)):
        return value * sf
    else:
        raise TypeError(f"Invalid type: {type(value)}")
