import json
from pathlib import Path
from typing import Literal
import numpy as np
from scipy.stats import truncnorm
from config.config import AVAILABLE_COMP, DISTR_TYPING


class Component:
    comp_data_path = Path('data/ATCCurves_json')

    def __init__(self,
                 ID: str):
        if not ID in AVAILABLE_COMP:
            raise ValueError(f'Component "{ID}" is not available')
        self.ID = ID
        self.comp_data = self._get_comp_data()
        self.damage_states = self._get_DSs()
    
    def _get_comp_data(self):
        with open(self.comp_data_path / f'{self.ID}.json', "r") as f:
            comp_data: dict = json.load(f)
        return comp_data

    def _get_DSs(self):
        damage_states: dict[str, str | list] = {
            'edp_type': None,  # EDP类型
            'median': [],  # 中值EDP
            'beta': [],  # 离差
        }
        edp_type: str = self.comp_data['FragilityCurve']['EDPType']['TypeName']
        damage_states['edp_type'] = edp_type
        DSs: list[dict] = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        for _, DS in enumerate(DSs):
            median = float(DS['Median'])
            beta = float(DS['Beta'])
            damage_states['median'].append(median)
            damage_states['beta'].append(beta)
        return damage_states

    def _get_cost(self,
            quantity: float,
            ds: int,
            is_random: bool = False
        ) -> float:
        """计算在给定损伤状态的情况下的构件修复成本

        Args:
            quantity (float): 数量(以考虑修复成本的规模效应)
            ds (int): 损伤状态序号，从1开始
            is_random (bool, optional): 是否考虑随机分布，默认False，即使用中值
        """
        DS_num = len(self.comp_data['FragilityCurve']['DamageStates']['DamageState'])
        if ds > DS_num:
            raise ValueError(f'Component {self.ID} has only {DS_num} damage states, but {ds} is given.')
        elif ds < 1:
            raise ValueError(f'Damage state should be a positive integer, but {ds} is given.')
        DS: dict = self.comp_data['FragilityCurve']['DamageStates']['DamageState'][ds-1]
        if 'DamageStates' in DS:
            # 存在相同等级的互斥损伤状态
            subDSs: list[dict] = DS['DamageStates']['DamageState']
            percents = []
            for subDS in subDSs:
                percents.append(float(subDS['Percent']))
            if is_random:
                # 根据percents随机选择一个子损伤状态
                subDS = np.random.choice(subDSs, p=percents)
            else:
                # 使用percents中较大概率数对应的子损伤状态
                subDS = subDSs[np.argmax(percents)]
            cost_cons: dict[str, str | float] = _extract_cost(subDS)
        else:
            cost_cons: dict[str, str | float] = _extract_cost(DS)
        cost = _get_prob_cost(
            cost_cons['LowerQuantity'],
            cost_cons['MaxAmount'],
            cost_cons['UpperQuantity'],
            cost_cons['MinAmount'],
            cost_cons['Uncertainty'],
            cost_cons['CurveType'],
            quantity,
            is_random
        )
        return cost
    
    
#     def _get_DSs(self):
#         """获取损伤状态，将FEMA P58数据库扁平化"""
#         damage_states = {
#             'edp_type': None,  # EDP类型
#             'median': [],  # 中值EDP
#             'beta': [],  # 离差
#             'use_casualty': [],  # 是否用于计算伤亡
#             'energy_type': [],  # 能量曲线分布类型
#             'energy_median': [],  # 能量中值
#             'energy_dispersion': [],  # 能量离差
#             'carbon_type': [],  # 碳排曲线分布类型
#             'carbon_median': [],  # 碳排中值
#             'carbon_dispersion': [],  # 碳排离差
#             'cost_lower_quantity': [],  # 损失量下限
#             'cost_max_amount': [],  # 损失金额上限
#             'cost_upper_quantity': [],  # 损失量上限
#             'cost_min_amount': [],  # 损失金额下限
#             'cost_uncertainty': [],  # 损失金额不确定性
#             'cost_type': [],  # 损失金额曲线分布类型
#             'time_lower_quantity': [],  # 时间下限
#             'time_max_amount': [],  # 时间上限
#             'time_upper_quantity': [],  # 时间上限
#             'time_min_amount': [],  # 时间下限
#             'time_uncertainty': [],  # 时间不确定性
#             'time_type': [],  # 时间曲线分布类型
#         }  # 如果有互斥损伤状态，则列表内再嵌套列表
#         edp_type: str = self.comp_data['FragilityCurve']['EDPType']['TypeName']
#         damage_states['edp_type'] = edp_type
#         DSs: list[dict] = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
#         for i, DS in enumerate(DSs):
#             if 'DamageStates' in DS:
#                 # 存在相同等级的互斥损伤状态
#                 subDS: list[dict] = DS['DamageStates']['DamageState']

#             else:

#                 damage_states['median'].append(median)
#                 damage_states['beta'].append(beta)
#                 damage_states['use_casualty'].append(use_casualty)
#                 damage_states['energy_type'].append(energy_type)
#                 damage_states['energy_median'].append(energy_median)
#                 damage_states['energy_dispersion'].append(energy_dispersion)
#                 damage_states['carbon_type'].append(carbon_type)
#                 damage_states['carbon_median'].append(carbon_median)
#                 damage_states['carbon_dispersion'].append(carbon_dispersion)
#                 damage_states['cost_lower_quantity'].append(cost_lower_quantity)
#                 damage_states['cost_max_amount'].append(cost_max_amount)
#                 damage_states['cost_upper_quantity'].append(cost_upper_quantity)
#                 damage_states['cost_min_amount'].append(cost_min_amount)
#                 damage_states['cost_uncertainty'].append(cost_uncertainty)
#                 damage_states['cost_type'].append(cost_type)
#                 damage_states['time_lower_quantity'].append(time_lower_quantity)
#                 damage_states['time_max_amount'].append(time_max_amount)
#                 damage_states['time_upper_quantity'].append(time_upper_quantity)
#                 damage_states['time_min_amount'].append(time_min_amount)
#                 damage_states['time_uncertainty'].append(time_uncertainty)
#                 damage_states['time_type'].append(time_type)


# def _parse_damage_state(DS: dict, is_subDS: Literal[False, 'unkown']):
#     """解析单个损伤状态"""
#     if not is_subDS:
#         median = float(DS['Median'])
#         beta = float(DS['Beta'])
#     if 'DamageStates' in DS:
#         # 存在相同等级的互斥损伤状态
#         subDSs: list[dict] = DS['DamageStates']['DamageState']
#         sub_data_ls = []
#         for subDS in subDSs:
#             sub_data_ls.append(_parse_damage_state(subDS, True))
#     cons_group: dict = DS['ConsequenceGroup']  # Consequence group
#     use_casualty: bool = cons_group['UseCasualty']  # True
#     energy_type: str = cons_group['EnergyCurveType']  # Normal
#     energy_median = float(cons_group['EnergyMedian'])
#     energy_dispersion = float(cons_group['EnergyDispersion'])
#     carbon_type: str = cons_group['CarbonCurveType']
#     carbon_median = float(cons_group['CarbonMedian'])
#     carbon_dispersion = float(cons_group['CarbonDispersion'])
#     cost_cons: dict = cons_group['CostConsequence']
#     cost_lower_quantity = float(cost_cons['LowerQuantity'])
#     cost_max_amount = float(cost_cons['MaxAmount'])
#     cost_upper_quantity = float(cost_cons['UpperQuantity'])
#     cost_min_amount = float(cost_cons['MinAmount'])
#     cost_uncertainty = float(cost_cons['Uncertainty'])
#     cost_type: str = cost_cons['CurveType']
#     time_cons: dict = cons_group['TimeConsequence']
#     time_lower_quantity = float(time_cons['LowerQuantity'])
#     time_max_amount = float(time_cons['MaxAmount'])
#     time_upper_quantity = float(time_cons['UpperQuantity'])
#     time_min_amount = float(time_cons['MinAmount'])
#     time_uncertainty = float(time_cons['Uncertainty'])
#     time_type: str = time_cons['CurveType']
#     data = (use_casualty, energy_type, energy_median, energy_dispersion,\
#         carbon_type, carbon_median, carbon_dispersion, cost_lower_quantity,\
#         cost_max_amount, cost_upper_quantity, cost_min_amount, cost_uncertainty, cost_type,\
#         time_lower_quantity, time_max_amount, time_upper_quantity, time_min_amount,\
#         time_uncertainty, time_type)
#     if is_subDS:
#         return data
#     else:
#         return median, beta, data

def _extract_cost(DS: dict):
    """提取修复损失"""
    cons_group: dict = DS['ConsequenceGroup']  # Consequence group
    cost_cons: dict[str, str | float] = cons_group['CostConsequence']
    for key, val in cost_cons.items():
        try:
            cost_cons[key] = float(val)
        except ValueError:
            pass
    return cost_cons

def _get_prob_cost(
    lower_quantity: float,
    max_amount: float,
    upper_quantity: float,
    min_amount: float,
    uncertainty: float,
    curve_type: DISTR_TYPING,
    quantity: float,
    is_random: bool,
):
    """计算概率修复成本

    Args:
        lower_quantity (float): 最小数量
        max_amount (float): 单价中值上限
        upper_quantity (float): 最大数量
        min_amount (float): 单价中值下限
        uncertainty (float): 不确定性
        curve_type (DISTR_TYPING): 概率分布类型
        quantity (float): 数量
        is_random (bool, optional): 是否考虑概率分布

    Returns:
        float: 修复成本
    """
    quantity = np.clip(quantity, lower_quantity, upper_quantity)
    median = max_amount - (max_amount - min_amount) * (
        (quantity - lower_quantity) / (upper_quantity - lower_quantity)
    )
    if is_random:
        if curve_type == 'Normal':
            std = uncertainty * median
            low_bound = median - 1.28155 * std  # 10%分位数
            high_bound = median + 1.28155 * std  # 90%分位数
            a = (low_bound - median) / std
            b = (high_bound - median) / std 
            unit_cost = truncnorm.rvs(a, b, loc=median, scale=std)  # 截断采样
        elif curve_type == 'LogNormal':
            log_median = np.log(median)
            log_low = log_median - 1.28155 * uncertainty
            log_high = log_median + 1.28155 * uncertainty
            a = (log_low - log_median) / uncertainty
            b = (log_high - log_median) / uncertainty
            log_unit_cost = truncnorm.rvs(a, b, loc=log_median, scale=uncertainty)
            unit_cost = np.exp(log_unit_cost)
    else:
        unit_cost = median
    return unit_cost * quantity