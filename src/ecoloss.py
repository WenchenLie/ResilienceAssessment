import os
import json
from pathlib import Path
import numpy as np
from multiprocessing import Pool, Manager
from threading import Thread
from typing import Literal
from .building import Building
from config.config import LOGGER, EDP_ABBR_TYPING


def intensity_based_loss(
    n: int,
    Sa_ls: np.ndarray,
    building: Building,
    output_dir: str | Path,
    is_random: bool = True,
    random_seed: int = None,
    parallel: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """基于地震动强度计算建筑的直接经济损失 (支持并行+进度监控)"""

    cost_mat_clps = np.zeros((len(Sa_ls), n))
    cost_mat_dm = np.zeros((len(Sa_ls), n))
    cost_mat_repair = np.zeros((len(Sa_ls), n))
    cost_mat_repair_category = {
        'S': np.zeros((len(Sa_ls), n)),
        'NS': np.zeros((len(Sa_ls), n)),
        'C': np.zeros((len(Sa_ls), n))
    }  # 3种构件分类(S, NS, C)
    cost_mat_repair_sensitivity = {
        'D': np.zeros((len(Sa_ls), n)),
        'ED': np.zeros((len(Sa_ls), n)),
        'A': np.zeros((len(Sa_ls), n)),
        'L': np.zeros((len(Sa_ls), n)),
        'LB': np.zeros((len(Sa_ls), n)),
        'V': np.zeros((len(Sa_ls), n))
    }  # 5种敏感性类型(D, ED, A, L, LB, V)

    if parallel <= 1:
        # 串行
        for idx_MC in range(n):
            print(f"  Running Monte Carlo simulation: {idx_MC+1}/{n}", end='\r')
            idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category, cost_repair_sensitivity\
                = _worker_wrapper(idx_MC, Sa_ls, building, is_random, random_seed, None)
            cost_mat_clps[:, idx_MC] = cost_clps
            cost_mat_dm[:, idx_MC] = cost_dm
            cost_mat_repair[:, idx_MC] = cost_repair
            for key, arr in cost_repair_category.items():
                cost_mat_repair_category[key][:, idx_MC] = arr
            for key, arr in cost_repair_sensitivity.items():
                cost_mat_repair_sensitivity[key][:, idx_MC] = arr
    else:
        # 多进程
        manager = Manager()
        queue = manager.Queue()

        def monitor():
            completed = 0
            while completed < n:
                queue.get()
                completed += 1
                print(f"  Running Monte Carlo simulation: {completed}/{n}", end='\r')
            LOGGER.success("Monte Carlo simulation completed.")

        monitor_thread = Thread(target=monitor)
        monitor_thread.start()
        with Pool(processes=parallel) as pool:
            args_list = [(idx_MC, Sa_ls, building, is_random, random_seed, queue) for idx_MC in range(n)]
            results = pool.starmap(_worker_wrapper, args_list)
            for idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category, cost_repair_sensitivity in results:
                cost_mat_clps[:, idx_MC] = cost_clps
                cost_mat_dm[:, idx_MC] = cost_dm
                cost_mat_repair[:, idx_MC] = cost_repair
                for key, arr in cost_repair_category.items():
                    cost_mat_repair_category[key][:, idx_MC] = arr
                for key, arr in cost_repair_sensitivity.items():
                    cost_mat_repair_sensitivity[key][:, idx_MC] = arr
        monitor_thread.join()

    # 基于建筑重建成本进行归一化
    cost_total = cost_mat_clps + cost_mat_dm + cost_mat_repair
    cost_total /= building.replacement_cost
    cost_mat_clps /= building.replacement_cost
    cost_mat_dm /= building.replacement_cost
    cost_mat_repair /= building.replacement_cost
    for key, arr in cost_repair_category.items():
        cost_mat_repair_category[key] /= building.replacement_cost
    for key, arr in cost_repair_sensitivity.items():
        cost_mat_repair_sensitivity[key] /= building.replacement_cost

    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
    # 所有保存的经济损失均为归一化数据
    np.save(output_dir / 'cost_total.npy', cost_total)
    np.save(output_dir / 'cost_collapse.npy', cost_mat_clps)
    np.save(output_dir / 'cost_demolishment.npy', cost_mat_dm)
    np.save(output_dir / 'cost_repair.npy', cost_mat_repair)
    for key, arr in cost_mat_repair_category.items():
        np.save(output_dir / f'cost_repair_category_{key}.npy', arr)
    for key, arr in cost_mat_repair_sensitivity.items():
        np.save(output_dir / f'cost_repair_sensitivity_{key}.npy', arr)
    np.save(output_dir / 'Sa.npy', Sa_ls)
    np.save(output_dir / 'replacement_cost.npy', building.replacement_cost)

    return cost_total, cost_mat_clps, cost_mat_dm, cost_mat_repair

def _worker_wrapper(
        idx_MC: int,
        Sa_ls: np.ndarray,
        building: Building,
        is_random: bool,
        random_seed: int | None,
        queue
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray,
               dict[str, np.ndarray], dict[str, np.ndarray]]:
    try:
        if random_seed is not None:
            np.random.seed(random_seed)
        cost_clps, cost_dm, cost_repair = np.zeros_like(Sa_ls), np.zeros_like(Sa_ls), np.zeros_like(Sa_ls)
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
                cost_clps[idx_Sa] = building.replacement_cost
                continue
            if building._simu_demolishment(RIDR, is_random):
                cost_dm[idx_Sa] = building.replacement_cost
                continue
            cost = 0
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
                ds_flag = comp._simu_DS(edp, is_random)
                if ds_flag == 0:
                    cost_i = 0
                else:
                    cost_i = comp._get_cost(quantity, ds_flag, is_random)
                cost += cost_i
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
    except Exception as e:
        LOGGER.error(f"Error in Monte Carlo simulation {idx_MC}: {e}")
        print(e)
        raise e
    if queue is not None:
        queue.put(1)
    return idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category, cost_repair_sensitivity

def time_based_loss(
        results_IBL: str | Path,
        hazard_curve: np.ndarray,
        output_dir: str | Path,
    ) -> tuple[float, float, float, float]:
    
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    
    results_IBL = Path(results_IBL)
    Sa_ls: np.ndarray = np.load(results_IBL / 'Sa.npy')
    cost_total: np.ndarray = np.load(results_IBL / 'cost_total.npy')
    cost_clps: np.ndarray = np.load(results_IBL / 'cost_collapse.npy')
    cost_dm: np.ndarray = np.load(results_IBL / 'cost_demolishment.npy')
    cost_repair: np.ndarray = np.load(results_IBL / 'cost_repair.npy')
    # replacement_cost: np.ndarray = np.load(results_IBL /'replacement_cost.npy')
    
    cost_total_mean, cost_total_median, cost_total_std =\
        np.mean(cost_total, axis=1), np.median(cost_total, axis=1), np.std(cost_total, axis=1)
    cost_clps_mean, cost_clps_median, cost_clps_std =\
        np.mean(cost_clps, axis=1), np.median(cost_clps, axis=1), np.std(cost_clps, axis=1)
    cost_dm_mean, cost_dm_median, cost_dm_std =\
        np.mean(cost_dm, axis=1), np.median(cost_dm, axis=1), np.std(cost_dm, axis=1)
    cost_repair_mean, cost_repair_median, cost_repair_std =\
        np.mean(cost_repair, axis=1), np.median(cost_repair, axis=1), np.std(cost_repair, axis=1)
    n_Sa, n_MC = cost_total.shape
    
    get_log10_harzard_curve = interp1d(np.log10(hazard_curve[:, 0]), np.log10(hazard_curve[:, 1]),
                                       kind='cubic', fill_value='extrapolate', bounds_error=False)
    x_hazard = np.linspace(hazard_curve[0, 0], hazard_curve[-1, 0], 1000)  # 仅用于画图
    y_hazard = pow(10, get_log10_harzard_curve(np.log10(x_hazard)))
    log10_HSa = get_log10_harzard_curve(np.log10(Sa_ls))  # 灾害曲线的对数
    HSa = np.power(10, log10_HSa)  # 灾害曲线纵坐标
    diff_HSa = np.append(0, np.diff(HSa))  # 灾害曲线的差分
    
    # 年度经济损失
    EAL_total = float(np.sum(cost_total_mean * np.abs(diff_HSa)))
    EAL_clps = float(np.sum(cost_clps_mean * np.abs(diff_HSa)))
    EAL_dm = float(np.sum(cost_dm_mean * np.abs(diff_HSa)))
    EAL_repair = float(np.sum(cost_repair_mean * np.abs(diff_HSa)))

    # 年度超越概率-经济损失曲线
    # P (L > l | IM = Sa) = ∫ P (L > l | IM = Sa) * d λ(e)
    loss_ls = np.arange(0.01, 1.01, 0.01)
    annual_rate = np.zeros_like(loss_ls)
    for idx_loss, loss in enumerate(loss_ls):
        # 建立地震强度Sa下的经济损失概率分布 P(L > l | IM = Sa)
        for idx_Sa, Sa in enumerate(Sa_ls):
            # 统计 cost_total[idx_Sa] 中大于等于 loss 的比例
            P = len(cost_total[idx_Sa][cost_total[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate[idx_loss] += P * d_lamda

    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
    data = {
        'EAL_total': EAL_total,
        'EAL_collapse': EAL_clps,
        'EAL_demolishment': EAL_dm,
        'EAL_repair': EAL_repair
    }

    json.dump(data, open(output_dir / 'Expected annual loss ratio.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    np.save(output_dir / 'USGS_hazard.npy', hazard_curve)
    np.save(output_dir / 'hazard_fitting.npy', np.column_stack((x_hazard, y_hazard)))
    np.save(output_dir / 'annual_rate.npy', np.column_stack((loss_ls, annual_rate)))

    return EAL_total, EAL_clps, EAL_dm, EAL_repair
