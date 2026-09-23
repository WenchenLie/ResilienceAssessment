import traceback
from typing import TypeVar, Literal
import numpy as np
from src.building import Building
from ._calculation import _normal
from config.config import EDP_ABBR_TYPING, LOGGER,\
    HOURS, MONTHS


def _realization(
        idx_MC: int,
        Sa_ls: np.ndarray,
        building: Building,
        is_random: bool,
        random_seed: int | None,
        queue
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray,
               dict[str, np.ndarray], dict[str, np.ndarray]]:
    """进行一次蒙特卡洛模拟计算consequence"""
    try:
        if random_seed is not None:
            np.random.seed(random_seed)
        # ↓ 由倒塌、拆除、修复导致的经济损失
        cost_clps, cost_dm, cost_repair = np.zeros_like(Sa_ls), np.zeros_like(Sa_ls), np.zeros_like(Sa_ls)
        repair_time = np.zeros_like(Sa_ls)  # 修复时间
        death_rate = np.zeros_like(Sa_ls)  # 死亡率
        injury_rate = np.zeros_like(Sa_ls)  # 受伤率
        cost_repair_category = {
            'S': np.zeros_like(Sa_ls),
            'NS': np.zeros_like(Sa_ls),
            'C': np.zeros_like(Sa_ls)
        }  # 不同类型(S, NS, C)的构件的修复成本，总和应等于cost_repair
        cost_repair_sensitivity = {
            'D': np.zeros_like(Sa_ls),
            'ED': np.zeros_like(Sa_ls),
            'A': np.zeros_like(Sa_ls),
            'L': np.zeros_like(Sa_ls),
            'LB': np.zeros_like(Sa_ls),
            'V': np.zeros_like(Sa_ls),
            'VED': np.zeros_like(Sa_ls),
        }  # 不同敏感性类型的构件的修复成本

        IDR_mat = np.zeros((len(Sa_ls), building.Nstory))
        maxRIDR_ls = np.zeros(len(Sa_ls))
        PFA_mat = np.zeros((len(Sa_ls), building.Nstory))
        flags: list[Literal['clps', 'dm', 'rp']] = []
        for idx_Sa, Sa in enumerate(Sa_ls):
            IDR = building._simu_IDR(Sa, is_random, idx_Sa, idx_MC)
            maxRIDR = building._simu_RIDR(Sa, is_random, idx_Sa, idx_MC)
            PFA = building._simu_PFA(Sa, is_random, idx_Sa, idx_MC)
            VED = building._simu_VED(Sa, is_random, idx_Sa, idx_MC)
            IDR_mat[idx_Sa] = IDR
            maxRIDR_ls[idx_Sa] = maxRIDR
            PFA_mat[idx_Sa] = PFA
            hour: TypeVar[HOURS] = np.random.choice(HOURS)
            month: TypeVar[HOURS] = np.random.choice(MONTHS)
            weekday: Literal[0, 1] = np.random.choice([0, 1], p=[5/7, 2/7])
            pop_num = building.pop_num * building.pop_day[hour][weekday] / 100 *\
                building.pop_month[month][weekday] / 100
            beta = building.pop_beta
            pop_num = _normal(pop_num, beta)  # 每1000sf的人数
            pop_num = max(pop_num, 1)  # 人数不能为0
            total_pop = pop_num * building.floor_area / 1000 * building.Nstory  # 总人口
            flag: Literal['rp', 'clps', 'dm'] = 'rp'

            if building._simu_clps(Sa, is_random):
                # 结构倒塌
                cost_clps[idx_Sa] = building.replacement_cost
                repair_time[idx_Sa] = building.replacement_time
                potential_clps_modes: list[tuple[int]] = list(building.collapse_modes.keys())
                p = list(building.collapse_modes.values())
                clps_mode_idx: int = np.random.choice(range(len(potential_clps_modes)), p=p)  # 倒塌模式
                clps_mode: tuple[int] = potential_clps_modes[clps_mode_idx]
                # 计算倒塌导致的人员伤亡(数量)
                death_, injury_ = 0, 0
                collapse_fatality_rate = _normal(building.collapse_fatality_rate, building.collapse_fatality_COV)
                collapse_injury_rate = _normal(building.collapse_injury_rate, building.collapse_injury_COV)
                collapse_fatality_rate = max(collapse_fatality_rate, 0)
                collapse_injury_rate = max(collapse_injury_rate, 0)
                for story in clps_mode:
                    # 倒塌的楼层
                    death_i = collapse_fatality_rate * pop_num * building.floor_area / 1000  # 死亡人数
                    injury_i = collapse_injury_rate * pop_num * building.floor_area / 1000  # 受伤人数
                    death_ += death_i
                    injury_ += injury_i
                death_rate[idx_Sa] = death_ / total_pop
                injury_rate[idx_Sa] = injury_ / total_pop
                flag = 'clps'
                # continue

            if building._simu_demolishment(maxRIDR, is_random):
                # 结构因残余变形过大而拆除
                if not flag == 'clps':
                    cost_dm[idx_Sa] = building.replacement_cost
                    repair_time[idx_Sa] = building.replacement_time
                flag = 'dm'
                # continue

            cost_, time_, death_, injury_ = 0, 0, 0, 0
            for comp, quantity, story, floor in building.components:
                # 遍历构件(每批)计算损失
                edp_type: EDP_ABBR_TYPING = comp.edp_type
                if edp_type == 'D':
                    edp = IDR[story - 1]
                elif edp_type == 'A':
                    if floor >= 2:
                        edp = PFA[floor - 2]
                    else:
                        edp = Sa
                elif edp_type == 'VED':
                    edp = VED[story - 1]
                else:
                    raise NotImplementedError(f'其他类型的EDP尚未实现: "{edp_type}"')
                ds_flag = comp._simu_DS(float(edp), is_random)  # 获取构件损伤状态
                cost_i = 0
                time_i = 0
                death_i = 0
                injury_i = 0
                if ds_flag == 0:
                    # 无损伤
                    pass
                else:
                    # 损伤
                    if flag == 'rp':
                        # 仅当可修复时才计算修复成本和时间
                        cost_i = comp._get_cost(quantity, ds_flag, is_random)  # 单个构件修复成本
                        time_i = comp._get_time(quantity, ds_flag, is_random)  # 单个构件修复时间
                    elif flag in ['rp', 'dm']:
                        # 仅当可修复或拆除时才计算伤亡
                        area, death_rate_, injury_rate_\
                            = comp._get_casualty(quantity, ds_flag, is_random)  # 该楼层受影响的总面积，死亡率，伤害率
                        death_i = area / 1000 * pop_num * death_rate_  # 死亡人数
                        injury_i = area / 1000 * pop_num * injury_rate_  # 受伤人数
                cost_ += cost_i
                time_ += time_i
                death_ += death_i / total_pop
                injury_ += injury_i / total_pop

                # 记录结果
                if flag == 'rp':
                    match comp.category:
                        case 'S':
                            cost_repair_category['S'][idx_Sa] += cost_i
                        case 'NS':
                            cost_repair_category['NS'][idx_Sa] += cost_i
                        case 'C':
                            cost_repair_category['C'][idx_Sa] += cost_i
                    match edp_type:
                        case 'D':
                            cost_repair_sensitivity['D'][idx_Sa] += cost_i
                        case 'ED':
                            cost_repair_sensitivity['ED'][idx_Sa] += cost_i
                        case 'A':
                            cost_repair_sensitivity['A'][idx_Sa] += cost_i
                        case 'L':
                            cost_repair_sensitivity['L'][idx_Sa] += cost_i
                        case 'LB':
                            cost_repair_sensitivity['LB'][idx_Sa] += cost_i
                        case 'V':
                            cost_repair_sensitivity['V'][idx_Sa] += cost_i
                        case 'VED':
                            cost_repair_sensitivity['VED'][idx_Sa] += cost_i
            if flag in 'rp':
                cost_repair[idx_Sa] = cost_
                repair_time[idx_Sa] = time_
            if flag in ['rp', 'dm']:
                death_rate[idx_Sa] = death_
                injury_rate[idx_Sa] = injury_
            flags.append(flag)
    except Exception as e:
        traceback.print_exc()
        LOGGER.error(f"Error in Monte Carlo simulation {idx_MC}: {e}")
        print(e)
        raise e
    if queue is not None:
        queue.put(1)
    return idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category, cost_repair_sensitivity,\
        repair_time, death_rate, injury_rate, IDR_mat, maxRIDR_ls, PFA_mat, flags

