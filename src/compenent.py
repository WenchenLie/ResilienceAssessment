import json
from pprint import pprint
from math import isclose
from pathlib import Path
from typing import Literal, Self
import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from config.config import AVAILABLE_COMP, DISTR_TYPING, EDP_TYPING


class Component:
    comp_data_path = Path('data/ATCCurves_json')

    def __init__(self,
            ID: str,
            show_info: bool = False,
            __user_defined: bool = False,
            __json_file: str | Path = None
        ):
        """定义一个构件，基于FEMA P58数据库

        Args:
            ID (str): 构件ID
            show_info (bool, optional): 是否打印主要信息
        """
        if not __user_defined:
            if not ID in AVAILABLE_COMP:
                raise ValueError(f'Component "{ID}" is not available')
            self.ID = ID
            self.comp_data = self._get_comp_data()
        else:
            self.comp_data: dict = json.load(open(__json_file, "r"))
            self.ID = self.comp_data['FragilityCurve']['ID']
        self.damage_states = self._get_DSs()
        if show_info:
            self.show_info()
    
    @classmethod
    def user_component(cls,
            json_file: str | Path,
            show_info: bool = False
        ) -> Self:
        """用户自定义一个构件

        Args:
            json_file (str | Path): 包含易损性信息的json文件，可参考`template.json`
            show_info (bool, optional): 是否打印主要信息
        """
        return cls(None, show_info, True, json_file)
    
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
        if isinstance(DSs, dict):
            DSs = [DSs]  # 只有一个损伤状态的情况
        for _, DS in enumerate(DSs):
            median = float(DS['Median'])
            beta = float(DS['Beta'])
            damage_states['median'].append(median)
            damage_states['beta'].append(beta)
        return damage_states

    def _simu_DS(self,
        edp: float,
        is_random: bool = False
    ) -> int:
        """在给定edp下，模拟构件的损伤状态

        Args:
            edp (float): _description_
            is_random (bool, optional): _description_. Defaults to False.

        Returns:
            int: 损伤状态序号，从0(无损伤)开始，1开始表示具体的损伤状态
        """
        median_ls: list[float] = self.damage_states['median']  # 各个损伤状态的edp中值
        ds_flag = [i for i in range(1, len(median_ls) + 1)]  # 各个损伤状态的序号
        beta_ls: list[float] = self.damage_states['beta']  # 各个损伤状态edp的beta
        prob = []  # 各个损伤状态的概率
        for i in range(len(ds_flag)):
            median, beta = median_ls[i], beta_ls[i]
            pi = norm.cdf(np.log((edp / median) / beta), 0, 1)  # 构件易损性曲线纵坐标值
            prob.append(pi)
        ds_flag.insert(0, 0)  # 添加无损伤状态
        prob.insert(0, 1)  # 无损伤状态的概率为1
        ds = zip(ds_flag, prob)  # 各个损伤状态的序号、超越概率
        ds = sorted(ds, key=lambda x: x[1], reverse=True)  # 按照prob降序排列
        ds_flag, prob = zip(*ds)
        prob1 = []  # 各个损伤状态的概率(包含无损伤)(总和=1)
        for i in range(len(ds_flag)):
            if i != len(ds_flag) - 1:
                prob1.append(prob[i] - prob[i + 1])
            else:
                prob1.append(prob[-1])
        if is_random:
            ds = np.random.choice(ds_flag, p=prob1)
        else:
            ds = ds_flag[np.argmax(prob1)]  # 取概率最大的损伤状态
        return ds
                

    def _get_cost(self,
            quantity: float,
            ds: int,
            is_random: bool = False
        ) -> float:
        """计算在给定损伤状态的情况下的构件修复成本

        Args:
            quantity (float): 数量
            ds (int): 损伤状态序号，从1开始
            is_random (bool, optional): 是否考虑随机分布，默认False，即使用中值
        
        Returns:
            float: 修复成本
        """
        DS_data = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if ds > len(DS_data):
            raise ValueError(f'Component {self.ID} has only {len(DS_data)} damage states, but {ds} is given.')
        elif ds < 1:
            raise ValueError(f'Damage state should be a positive integer, but {ds} is given.')
        DSs: dict = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if isinstance(DSs, dict):
            DS = DSs
        else:
            DS = DSs[ds-1]
        if 'DamageStates' in DS:
            # 存在相同等级的互斥损伤状态
            subDSs: list[dict] = DS['DamageStates']['DamageState']
            DSGroupType: str = DS['DamageStates']['DSGroupType']
            if DSGroupType == 'MutuallyExclusive':
                # 互斥的子损伤状态
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
            elif DSGroupType == 'Simultaneous':
                # 并行的子损伤状态
                cost = 0
                for subDS in subDSs:
                    percent = float(subDS['Percent'])
                    cost_cons = _extract_cost(subDS)
                    cost_i = _get_prob_cons(
                        cost_cons['LowerQuantity'],
                        cost_cons['MaxAmount'],
                        cost_cons['UpperQuantity'],
                        cost_cons['MinAmount'],
                        cost_cons['Uncertainty'],
                        cost_cons['CurveType'],
                        quantity,
                        is_random
                    )
                    if is_random:
                        cost_i = np.random.choice([cost_i, 0], p=[percent, 1-percent])
                    else:
                        cost_i = percent * cost_i
                    cost += cost_i
                return cost
        else:
            cost_cons: dict[str, str | float] = _extract_cost(DS)
        cost = _get_prob_cons(
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
    
    def _get_time(self,
            quantity: float,
            ds: int,
            is_random: bool = False
        ) -> float:
        """计算在给定损伤状态的情况下的构件修复时间

        Args:
            quantity (float): 数量
            ds (int): 损伤状态序号，从1开始
            is_random (bool, optional): 是否考虑随机分布，默认False，即使用中值
        
        Returns:
            float: 修复时间
        """
        DS_data = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if ds > len(DS_data):
            raise ValueError(f'Component {self.ID} has only {len(DS_data)} damage states, but {ds} is given.')
        elif ds < 1:
            raise ValueError(f'Damage state should be a positive integer, but {ds} is given.')
        DSs: dict = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if isinstance(DSs, dict):
            DS = DSs
        else:
            DS = DSs[ds-1]
        if 'DamageStates' in DS:
            # 存在相同等级的互斥损伤状态
            subDSs: list[dict] = DS['DamageStates']['DamageState']
            DSGroupType: str = DS['DamageStates']['DSGroupType']
            if DSGroupType == 'MutuallyExclusive':
                # 互斥的子损伤状态
                percents = []
                for subDS in subDSs:
                    percents.append(float(subDS['Percent']))
                if is_random:
                    # 根据percents随机选择一个子损伤状态
                    subDS = np.random.choice(subDSs, p=percents)
                else:
                    # 使用percents中较大概率数对应的子损伤状态
                    subDS = subDSs[np.argmax(percents)]
                time_cons: dict[str, str | float] = _extract_time(subDS)
            elif DSGroupType == 'Simultaneous':
                # 并行的子损伤状态
                time = 0
                for subDS in subDSs:
                    percent = float(subDS['Percent'])
                    time_cons = _extract_time(subDS)
                    time_i = _get_prob_cons(
                        time_cons['LowerQuantity'],
                        time_cons['MaxAmount'],
                        time_cons['UpperQuantity'],
                        time_cons['MinAmount'],
                        time_cons['Uncertainty'],
                        time_cons['CurveType'],
                        quantity,
                        is_random
                    )
                    if is_random:
                        time_i = np.random.choice([time_i, 0], p=[percent, 1-percent])
                    else:
                        time_i = percent * time_i
                    time += time_i
                return time
        else:
            time_cons: dict[str, str | float] = _extract_time(DS)
        time = _get_prob_cons(
            time_cons['LowerQuantity'],
            time_cons['MaxAmount'],
            time_cons['UpperQuantity'],
            time_cons['MinAmount'],
            time_cons['Uncertainty'],
            time_cons['CurveType'],
            quantity,
            is_random
        )
        return time

    def _get_energy(self,
            quantity: float,
            ds: int,
            is_random: bool = False
        ) -> float:
        """计算在给定损伤状态的情况下的构件修复的能量消耗

        Args:
            quantity (float): 数量
            ds (int): 损伤状态序号，从1开始
            is_random (bool, optional): 是否考虑随机分布，默认False，即使用中值

        Returns:
            float: 修复的能量消耗
        """
        DS_data = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if ds > len(DS_data):
            raise ValueError(f'Component {self.ID} has only {len(DS_data)} damage states, but {ds} is given.')
        elif ds < 1:
            raise ValueError(f'Damage state should be a positive integer, but {ds} is given.')
        DSs: dict = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if isinstance(DSs, dict):
            DS = DSs
        else:
            DS = DSs[ds-1]
        if 'DamageStates' in DS:
            # 存在相同等级的互斥损伤状态
            subDSs: list[dict] = DS['DamageStates']['DamageState']
            DSGroupType: str = DS['DamageStates']['DSGroupType']
            if DSGroupType == 'MutuallyExclusive':
                # 互斥的子损伤状态
                percents = []
                for subDS in subDSs:
                    percents.append(float(subDS['Percent']))
                if is_random:
                    # 根据percents随机选择一个子损伤状态
                    subDS = np.random.choice(subDSs, p=percents)
                else:
                    # 使用percents中较大概率数对应的子损伤状态
                    subDS = subDSs[np.argmax(percents)]
                energy_median, energy_type, energy_dispersion = _extract_energy(subDS)
            elif DSGroupType == 'Simultaneous':
                # 并行的子损伤状态
                energy = 0
                for subDS in subDSs:
                    percent = float(subDS['Percent'])
                    energy_median, energy_type, energy_dispersion = _extract_energy(subDS)
                    energy_i = _get_prob_cons(1, energy_median, 2, energy_median, energy_dispersion,
                                              energy_type, quantity, is_random)
                    if is_random:
                        energy_i = np.random.choice([energy_i, 0], p=[percent, 1-percent])
                    else:
                        energy_i = percent * energy_i
                    energy += energy_i
                return energy
        else:
            energy_median, energy_type, energy_dispersion = _extract_energy(DS)
        energy = _get_prob_cons(1, energy_median, 2, energy_median, energy_dispersion,
                              energy_type, quantity, is_random)
        return energy

    def _get_carbon(self,
            quantity: float,
            ds: int,
            is_random: bool = False
        ) -> float:
        """计算在给定损伤状态的情况下的构件修复的能量消耗

        Args:
            quantity (float): 数量
            ds (int): 损伤状态序号，从1开始
            is_random (bool, optional): 是否考虑随机分布，默认False，即使用中值

        Returns:
            float: 碳排放
        """
        DS_data = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if ds > len(DS_data):
            raise ValueError(f'Component {self.ID} has only {len(DS_data)} damage states, but {ds} is given.')
        elif ds < 1:
            raise ValueError(f'Damage state should be a positive integer, but {ds} is given.')
        DSs: dict = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if isinstance(DSs, dict):
            DS = DSs
        else:
            DS = DSs[ds-1]
        if 'DamageStates' in DS:
            # 存在相同等级的互斥损伤状态
            subDSs: list[dict] = DS['DamageStates']['DamageState']
            DSGroupType: str = DS['DamageStates']['DSGroupType']
            if DSGroupType == 'MutuallyExclusive':
                # 互斥的子损伤状态
                percents = []
                for subDS in subDSs:
                    percents.append(float(subDS['Percent']))
                if is_random:
                    # 根据percents随机选择一个子损伤状态
                    subDS = np.random.choice(subDSs, p=percents)
                else:
                    # 使用percents中较大概率数对应的子损伤状态
                    subDS = subDSs[np.argmax(percents)]
                carbon_median, carbon_type, carbon_dispersion = _extract_carbon(subDS)
            elif DSGroupType == 'Simultaneous':
                # 并行的子损伤状态
                carbon = 0
                for subDS in subDSs:
                    percent = float(subDS['Percent'])
                    carbon_median, carbon_type, carbon_dispersion = _extract_carbon(subDS)
                    carbon_i = _get_prob_cons(1, carbon_median, 2, carbon_median, carbon_dispersion,
                                              carbon_type, quantity, is_random)
                    if is_random:
                        carbon_i = np.random.choice([carbon_i, 0], p=[percent, 1-percent])
                    else:
                        carbon_i = percent * carbon_i
                    carbon += carbon_i
                return carbon
        else:
            carbon_median, carbon_type, carbon_dispersion = _extract_carbon(DS)
        carbon = _get_prob_cons(1, carbon_median, 2, carbon_median, carbon_dispersion,
                              carbon_type, quantity, is_random)
        return carbon

    def _get_casualty(self,
            quantity: float,
            ds: int,
            is_random: bool = False
        ) -> tuple[float, float, float]:
        """计算在给定损伤状态的情况下的构件造成的死亡人数和严重受伤人数

        Args:
            quantity (float): 数量
            ds (int): 损伤状态序号，从1开始
            is_random (bool, optional): 是否考虑随机分布，默认False，即使用中值

        Returns:
            tuple[float, float, float]: 受影响的总面积，死亡率，伤亡率
        """
        DS_data = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if ds > len(DS_data):
            raise ValueError(f'Component {self.ID} has only {len(DS_data)} damage states, but {ds} is given.')
        elif ds < 1:
            raise ValueError(f'Damage state should be a positive integer, but {ds} is given.')
        DSs: dict = self.comp_data['FragilityCurve']['DamageStates']['DamageState']
        if isinstance(DSs, dict):
            DS = DSs
        else:
            DS = DSs[ds-1]
        if 'DamageStates' in DS:
            # 存在相同等级的互斥损伤状态
            subDSs: list[dict] = DS['DamageStates']['DamageState']
            DSGroupType: str = DS['DamageStates']['DSGroupType']
            if DSGroupType == 'MutuallyExclusive':
                # 互斥的子损伤状态
                percents = []
                for subDS in subDSs:
                    percents.append(float(subDS['Percent']))
                if is_random:
                    # 根据percents随机选择一个子损伤状态
                    subDS = np.random.choice(subDSs, p=percents)
                else:
                    # 使用percents中较大概率数对应的子损伤状态
                    subDS = subDSs[np.argmax(percents)]
                use_casualty, area, death_rate, death_rate_beta, injury_rate, injury_rate_beta = _extract_casualty(subDS)
            elif DSGroupType == 'Simultaneous':
                # 并行的子损伤状态
                death_rate, injury_rate = 0, 0
                for subDS in subDSs:
                    percent = float(subDS['Percent'])
                    use_casualty, area, death_rate, death_rate_beta, injury_rate, injury_rate_beta = _extract_casualty(subDS)
                    if not use_casualty:
                        return area * quantity, 0, 0
                    death_rate_i = _get_prob_cons(1, death_rate, 2, death_rate, death_rate_beta,
                                                'Normal', 1, is_random)
                    injury_rate_i = _get_prob_cons(1, injury_rate, 2, injury_rate, injury_rate_beta,
                                                'Normal', 1, is_random)
                    if is_random:
                        (death_rate_i, injury_rate_i) = np.random.choice([(death_rate_i, injury_rate_i), (0, 0)], p=[percent, 1-percent])
                    else:
                        death_rate_i, injury_rate_i = percent * death_rate_i, percent * injury_rate_i
                    death_rate += death_rate_i
                    injury_rate += injury_rate_i
                return area * quantity, death_rate, injury_rate
        else:
            use_casualty, area, death_rate, death_rate_beta, injury_rate, injury_rate_beta = _extract_casualty(DS)
        if not use_casualty:
            return area * quantity, 0, 0
        death_rate = _get_prob_cons(1, death_rate, 2, death_rate, death_rate_beta,
                                    'Normal', 1, is_random)
        injury_rate = _get_prob_cons(1, injury_rate, 2, injury_rate, injury_rate_beta,
                                    'Normal', 1, is_random)
        return area * quantity, death_rate, injury_rate
  
    def show_info(self, is_print: bool = True) -> dict[str, str]:
        """获取构件关键信息

        Args:
            is_print (bool, optional): 是否打印，默认True
        
        Returns:
            dict[str, str]: 构件关键信息
        """
        data = self.comp_data['FragilityCurve']
        info = {
            'ID': self.ID,
            'component_name': data['Name'],
            'description': data['Directional'],
            'EDP_type': data['EDPType']["TypeName"],
            'EDP_unit': data['EDPType']["DefaultUnits"],
        }
        if is_print:
            pprint(info)
        return info



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

def _extract_time(DS: dict):
    """提取修复时间"""
    cons_group: dict = DS['ConsequenceGroup']  # Consequence group
    time_cons: dict[str, str | float] = cons_group['TimeConsequence']
    for key, val in time_cons.items():
        try:
            time_cons[key] = float(val)
        except ValueError:
            pass
    return time_cons

def _extract_energy(DS: dict):
    """提取修复能量消耗"""
    cons_group: dict = DS['ConsequenceGroup']  # Consequence group
    energy_median = float(cons_group['EnergyMedian'])
    energy_type: DISTR_TYPING = cons_group['EnergyCurveType']
    energy_dispersion = float(cons_group['EnergyDispersion'])
    return energy_median, energy_type, energy_dispersion

def _extract_carbon(DS: dict):
    """提取碳排放"""
    cons_group: dict = DS['ConsequenceGroup']  # Consequence group
    carbon_median = float(cons_group['CarbonMedian'])
    carbon_type: DISTR_TYPING = cons_group['CarbonCurveType']
    carbon_dispersion = float(cons_group['CarbonDispersion'])
    return carbon_median, carbon_type, carbon_dispersion

def _extract_casualty(DS: dict):
    """提取伤亡信息"""
    cons_group: dict = DS['ConsequenceGroup']  # Consequence group
    use_casualty: bool = cons_group['UseCasualty']
    area = cons_group['AffectedFloorArea']
    if area is None:
        area = 0
    else:
        area = float(area['Area']['Value'])  # 单位恒为 Square Foot
    death_rate: float = float(cons_group['AffectedDeathRate'])
    death_rate_beta = float(cons_group['AffectedDeathRateBeta'])
    injury_rate: float = float(cons_group['AffectedInjuryRate'])
    injury_rate_beta = float(cons_group['AffectedInjuryRateBeta'])
    return use_casualty, area, death_rate, death_rate_beta, injury_rate, injury_rate_beta

def _get_prob_cons(
    lower_quantity: float,
    max_amount: float,
    upper_quantity: float,
    min_amount: float,
    uncertainty: float,
    curve_type: DISTR_TYPING,
    quantity: float,
    is_random: bool,
) -> float:
    """计算概率修复成本和时间成本，考虑规模效应

    Args:
        lower_quantity (float): 最小数量
        max_amount (float): 单价/修复时间中值上限
        upper_quantity (float): 最大数量
        min_amount (float): 单价/修复时间中值下限
        uncertainty (float): 不确定性
        curve_type (DISTR_TYPING): 概率分布类型
        quantity (float): 数量
        is_random (bool, optional): 是否考虑概率分布

    Returns:
        float: 修复成本
    """
    if quantity < lower_quantity:
        median = max_amount
    elif quantity > upper_quantity:
        median = min_amount
    else:
        median = max_amount - (max_amount - min_amount) * (
            (quantity - lower_quantity) / (upper_quantity - lower_quantity)
        )
    if isclose(median, 0):
        return 0.0
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