import numpy as np
from tqdm import trange
from .building import Building
from config.config import EDP_TYPING, LOGGER


def intensity_based_loss(
    n: int,
    Sa_ls: np.ndarray,
    building: Building,
    is_random: bool = True,
    random_seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于地震动强度计算建筑的直接经济损失

    Args:
        n (int): 蒙特卡洛模拟次数
        Sa_ls (np.ndarray): 地震动强度
        building (Building): Building对象
        is_random (bool, optional): 是否考虑随机分布
        random_seed (int, optional): 随机种子，为None时不设置随机种子

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 分别由倒塌、拆除、修复导致的经济损失矩阵
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    cost_mat_clps = np.zeros((len(Sa_ls), n))  # 倒塌导致的经济损失
    cost_mat_dm = np.zeros((len(Sa_ls), n))  # 拆除导致的经济损失
    cost_mat_repair = np.zeros((len(Sa_ls), n))  # 修复经济损失
    for idx_MC in trange(n, desc='Monte Carlo Simulation'):
        for idx_Sa, Sa in (enumerate(Sa_ls)):
            IDR: np.ndarray = building._simu_IDR(Sa, is_random)
            RIDR: float = building._simu_RIDR(Sa, is_random)
            PFA: np.ndarray = building._simu_PFA(Sa, is_random)
            # PFV: np.ndarray = building._simu_PFV(Sa, is_random)
            if building._simu_clps(Sa, is_random):
                cost_mat_clps[idx_Sa, idx_MC] = building.replacement_cost
                continue
            if building._simu_demolishment(RIDR, is_random):
                cost_mat_dm[idx_Sa, idx_MC] = building.replacement_cost
                continue
            # 可修复，进行损失计算
            cost = 0  # 建筑总修复成本
            for comp, quantity, story, floor in building.components:
                edp_type: EDP_TYPING = comp.damage_states['edp_type']
                if edp_type == 'Story Drift Ratio':
                    edp = IDR[story - 1]
                elif edp_type == 'Acceleration':
                    if floor >= 2:
                        edp = PFA[floor - 2]
                    else:
                        edp = Sa
                else:
                    assert False, '其他类型的EDP尚未实现'
                ds_flag = comp._simu_DS(edp, is_random)  # 损伤状态标志
                if ds_flag == 0:
                    cost_i = 0  # 无损伤，无修复成本
                else:
                    cost_i = comp._get_cost(quantity, ds_flag, is_random)  # 修复成本
                cost += cost_i
            cost_mat_repair[idx_Sa, idx_MC] = cost
    cost_total = cost_mat_clps + cost_mat_dm + cost_mat_repair  # 总经济损失
    cost_total /= building.replacement_cost  # 损失归一化
    cost_mat_clps /= building.replacement_cost
    cost_mat_dm /= building.replacement_cost
    cost_mat_repair /= building.replacement_cost
    return cost_total, cost_mat_clps, cost_mat_dm, cost_mat_repair





def time_based_loss():
    ...

