import os
import json
from pathlib import Path
import numpy as np
from multiprocessing import Pool, Manager
from threading import Thread
from .building import Building
from ._realization import _realization
from config.config import LOGGER


def consequance_estimate(
    n: int,
    Sa_ls: np.ndarray,
    building: Building,
    hazard_curve: np.ndarray,
    root: str | Path,
    is_random: bool = True,
    random_seed: int = None,
    parallel: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算建筑的损失，包括修复成本、修复时间

    Args:
        n (int): 蒙特卡洛模拟次数
        Sa_ls (np.ndarray): 地震动强度
        building (Building): Building实例
        root (str | Path): 输出文件夹路径
        is_random (bool, optional): 是否考虑随机分布
        random_seed (int, optional): 随机数种子
        parallel (int, optional): 并行计算的进程数，默认为1，即串行

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    root = Path(root)
    if not root.exists():
        os.makedirs(root)
    cost_mat_clps = np.zeros((len(Sa_ls), n))
    cost_mat_dm = np.zeros((len(Sa_ls), n))
    cost_mat_repair = np.zeros((len(Sa_ls), n))
    repair_time_mat = np.zeros((len(Sa_ls), n))
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
            idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category, cost_repair_sensitivity,\
                repair_time = _realization(idx_MC, Sa_ls, building, is_random, random_seed, None)
            cost_mat_clps[:, idx_MC] = cost_clps
            cost_mat_dm[:, idx_MC] = cost_dm
            cost_mat_repair[:, idx_MC] = cost_repair
            repair_time_mat[:, idx_MC] = repair_time
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
            results = pool.starmap(_realization, args_list)
            for idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category, cost_repair_sensitivity,\
                repair_time in results:
                cost_mat_clps[:, idx_MC] = cost_clps
                cost_mat_dm[:, idx_MC] = cost_dm
                cost_mat_repair[:, idx_MC] = cost_repair
                repair_time_mat[:, idx_MC] = repair_time
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

    # 保存结果
    np.savetxt(root / 'Sa.txt', Sa_ls)
    np.savetxt(root / 'replacement_cost.txt', np.array([building.replacement_cost]))
    np.savetxt(root / 'replacement_time.txt', np.array([building.replacement_time]))
    # 修复成本
    output_dir = root / 'Repair cost'
    if not output_dir.exists():
        os.makedirs(output_dir)
    np.save(output_dir / 'cost_total.npy', cost_total)
    np.save(output_dir / 'cost_collapse.npy', cost_mat_clps)
    np.save(output_dir / 'cost_demolishment.npy', cost_mat_dm)
    np.save(output_dir / 'cost_repair.npy', cost_mat_repair)
    _time_based_loss(Sa_ls, cost_total, cost_mat_clps, cost_mat_dm, cost_mat_repair,
                     hazard_curve, root, output_dir)
    for key, arr in cost_mat_repair_category.items():
        np.save(output_dir / f'cost_repair_category_{key}.npy', arr)
    for key, arr in cost_mat_repair_sensitivity.items():
        np.save(output_dir / f'cost_repair_sensitivity_{key}.npy', arr)
    # 修复时间
    output_dir = root / 'Repair time'
    if not output_dir.exists():
        os.makedirs(output_dir)
    np.save(output_dir /'repair_time.npy', repair_time_mat)
    

def _time_based_loss(
        Sa_ls: np.ndarray,
        cost_total: np.ndarray,
        cost_clps: np.ndarray,
        cost_dm: np.ndarray,
        cost_repair: np.ndarray,
        hazard_curve: np.ndarray,
        root: Path,
        output_dir: Path
    ) -> tuple[float, float, float, float]:
    
    from scipy.interpolate import interp1d

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

    data = {
        'EAL_total': EAL_total,
        'EAL_collapse': EAL_clps,
        'EAL_demolishment': EAL_dm,
        'EAL_repair': EAL_repair
    }

    json.dump(data, open(output_dir / 'Expected annual loss ratio.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    np.savetxt(root / 'Hazard_USGS.txt', hazard_curve)
    np.savetxt(root / 'Hazard_fitting.txt', np.column_stack((x_hazard, y_hazard)))
    np.save(output_dir / 'annual_rate.npy', np.column_stack((loss_ls, annual_rate)))
