import numpy as np
from src.building import Building
from config.config import EDP_ABBR_TYPING, LOGGER


def _realization(
        idx_MC: int,
        Sa_ls: np.ndarray,
        building: Building,
        is_random: bool,
        random_seed: int | None,
        queue
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray,
               dict[str, np.ndarray], dict[str, np.ndarray]]:
    """对应于一次蒙特卡洛模拟"""
    try:
        if random_seed is not None:
            np.random.seed(random_seed)
        # ↓ 由倒塌、拆除、修复导致的经济损失
        cost_clps, cost_dm, cost_repair = np.zeros_like(Sa_ls), np.zeros_like(Sa_ls), np.zeros_like(Sa_ls)
        repair_time = np.zeros_like(Sa_ls)  # 修复时间
        death = np.zeros_like(Sa_ls)  # 死亡人数
        injury = np.zeros_like(Sa_ls)  # 受伤人数
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
            'V': np.zeros_like(Sa_ls)
        }  # 不同敏感性类型的构件的修复成本
        for idx_Sa, Sa in enumerate(Sa_ls):
            IDR = building._simu_IDR(Sa, is_random)
            RIDR = building._simu_RIDR(Sa, is_random)
            PFA = building._simu_PFA(Sa, is_random)
            if building._simu_clps(Sa, is_random):
                # 结构倒塌
                cost_clps[idx_Sa] = building.replacement_cost
                repair_time[idx_Sa] = building.replacement_time
                continue
            if building._simu_demolishment(RIDR, is_random):
                # 结构因残余变形过大而拆除
                cost_dm[idx_Sa] = building.replacement_cost
                repair_time[idx_Sa] = building.replacement_time
                continue
            # 结构可修复
            cost = 0
            time_ = 0
            for comp, quantity, story, floor in building.components:
                edp_type: EDP_ABBR_TYPING = comp.edp_type
                if edp_type == 'D':
                    edp = IDR[story - 1]
                elif edp_type == 'A':
                    if floor >= 2:
                        edp = PFA[floor - 2]
                    else:
                        edp = Sa
                else:
                    raise NotImplementedError(f'其他类型的EDP尚未实现: "{edp_type}"')
                ds_flag = comp._simu_DS(edp, is_random)  # 获取构件损伤状态
                if ds_flag == 0:
                    # 无损伤
                    cost_i = 0
                    time_i = 0
                    death_i = 0
                    injury_i = 0
                else:
                    # 损伤
                    cost_i = comp._get_cost(quantity, ds_flag, is_random)  # 单个构件修复成本
                    time_i = comp._get_time(quantity, ds_flag, is_random)  # 单个构件修复时间
                    area, death_rate, injury_rate\
                        = comp._get_casualty(quantity, ds_flag, is_random)  # 该楼层受影响的总面积，死亡率，伤害率
                cost += cost_i
                time_ += time_i
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
            cost_repair[idx_Sa] = cost
            repair_time[idx_Sa] = time_
    except Exception as e:
        LOGGER.error(f"Error in Monte Carlo simulation {idx_MC}: {e}")
        print(e)
        raise e
    if queue is not None:
        queue.put(1)
    return idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category, cost_repair_sensitivity,\
        repair_time

