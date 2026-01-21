import os
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
from threading import Thread
from .building import Building
from ._realization import _realization
from config.config import LOGGER


def consequance_estimate(
    building: Building,
    hazard_curve: np.ndarray,
    root: str | Path,
    is_random: bool = True,
    random_seed: int = None,
    parallel: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算建筑的损失，包括修复成本、修复时间

    Args:
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
    n = building.n_MC
    Sa_ls = building.Sa_ls
    cost_mat_clps = np.zeros((len(Sa_ls), n))
    cost_mat_dm = np.zeros((len(Sa_ls), n))
    cost_mat_repair = np.zeros((len(Sa_ls), n))
    repair_time_mat = np.zeros((len(Sa_ls), n))
    death_rate_mat = np.zeros((len(Sa_ls), n))
    injury_rate_mat = np.zeros((len(Sa_ls), n))
    IDR = np.zeros((len(Sa_ls), building.Nstory, n))
    maxRIDR = np.zeros((len(Sa_ls), n))
    PFA = np.zeros((len(Sa_ls), building.Nstory, n))
    flag_mat = np.zeros((len(Sa_ls), n), dtype=object)
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
            idx_MC, cost_clps, cost_dm, cost_repair, cost_repair_category,\
                cost_repair_sensitivity, repair_time, death_rate, injury_rate,\
                IDR_mat, maxRIDR_ls, PFA_mat, flags =\
                    _realization(idx_MC, Sa_ls, building, is_random, random_seed, None)
            # IDR_mat.shape = (len(Sa_ls), Nstory)
            cost_mat_clps[:, idx_MC] = cost_clps
            cost_mat_dm[:, idx_MC] = cost_dm
            cost_mat_repair[:, idx_MC] = cost_repair
            repair_time_mat[:, idx_MC] = repair_time
            death_rate_mat[:, idx_MC] = death_rate
            injury_rate_mat[:, idx_MC] = injury_rate
            IDR[:, :, idx_MC] = IDR_mat
            maxRIDR[:, idx_MC] = maxRIDR_ls
            PFA[:, :, idx_MC] = PFA_mat
            flag_mat[:, idx_MC] = flags
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
                repair_time, death_rate, injury_rate, IDR_mat, maxRIDR_ls, PFA_mat, flags in results:
                cost_mat_clps[:, idx_MC] = cost_clps
                cost_mat_dm[:, idx_MC] = cost_dm
                cost_mat_repair[:, idx_MC] = cost_repair
                repair_time_mat[:, idx_MC] = repair_time
                death_rate_mat[:, idx_MC] = death_rate
                injury_rate_mat[:, idx_MC] = injury_rate
                IDR[:, :, idx_MC] = IDR_mat
                maxRIDR[:, idx_MC] = maxRIDR_ls
                PFA[:, :, idx_MC] = PFA_mat
                flag_mat[:, idx_MC] = flags
                for key, arr in cost_repair_category.items():
                    cost_mat_repair_category[key][:, idx_MC] = arr
                for key, arr in cost_repair_sensitivity.items():
                    cost_mat_repair_sensitivity[key][:, idx_MC] = arr
        monitor_thread.join()
    IDR_median = np.median(IDR, axis=2)
    maxRIDR_median = np.median(maxRIDR, axis=1)
    PFA_median = np.median(PFA, axis=2)

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
    cols = ['Sa']+list(range(1, building.Nstory + 1))
    df_IDR_median = pd.DataFrame(np.column_stack((Sa_ls, IDR_median)), columns=cols)
    df_maxRIDR_median = pd.DataFrame(np.column_stack((Sa_ls, maxRIDR_median)), columns=['Sa', 'maxRIDR'])
    df_PFA_median = pd.DataFrame(np.column_stack((Sa_ls, PFA_median)), columns=cols)
    df_flags = pd.DataFrame(np.column_stack((Sa_ls, flag_mat)), columns=['Sa']+list(range(1, building.n_MC + 1)))
    df_IDR_median.to_csv(root / 'IDR_median.csv', index=False)
    df_maxRIDR_median.to_csv(root / 'maxRIDR_median.csv', index=False)
    df_PFA_median.to_csv(root / 'PFA_median.csv', index=False)
    df_flags.to_csv(root / 'flags.csv', index=False)
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
                     cost_mat_repair_category, cost_mat_repair_sensitivity, hazard_curve, root, output_dir)
    for key, arr in cost_mat_repair_category.items():
        np.save(output_dir / f'cost_repair_category_{key}.npy', arr)
    for key, arr in cost_mat_repair_sensitivity.items():
        np.save(output_dir / f'cost_repair_sensitivity_{key}.npy', arr)
    # 修复时间
    output_dir = root / 'Repair time'
    if not output_dir.exists():
        os.makedirs(output_dir)
    np.save(output_dir /'repair_time.npy', repair_time_mat)
    # 人员伤亡
    output_dir = root / 'Casualties'
    if not output_dir.exists():
        os.makedirs(output_dir)
    np.save(output_dir / 'Death_rate.npy', death_rate_mat)
    np.save(output_dir / 'Injury_rate.npy', injury_rate_mat)
    

def _time_based_loss(
        Sa_ls: np.ndarray,
        cost_total: np.ndarray,
        cost_clps: np.ndarray,
        cost_dm: np.ndarray,
        cost_repair: np.ndarray,
        cost_mat_repair_category: dict[str, np.ndarray],
        cost_mat_repair_sensitivity: dict[str, np.ndarray],
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
    cost_repair_S = cost_mat_repair_category['S']
    cost_repair_NS = cost_mat_repair_category['NS']
    cost_repair_C = cost_mat_repair_category['C']
    cost_repair_D = cost_mat_repair_sensitivity['D']
    cost_repair_ED = cost_mat_repair_sensitivity['ED']
    cost_repair_A = cost_mat_repair_sensitivity['A']
    cost_repair_L = cost_mat_repair_sensitivity['L']
    cost_repair_LB = cost_mat_repair_sensitivity['LB']
    cost_repair_V = cost_mat_repair_sensitivity['V']
    cost_repair_NS_S = cost_repair_NS - cost_repair_A  # 位移型非结构构件的修复成本，该式不是所有情况都成立
    cost_repair_S_mean, cost_repair_S_median, cost_repair_S_std =\
        np.mean(cost_repair_S, axis=1), np.median(cost_repair_S, axis=1), np.std(cost_repair_S, axis=1)
    cost_repair_NS_mean, cost_repair_NS_median, cost_repair_NS_std =\
        np.mean(cost_repair_NS, axis=1), np.median(cost_repair_NS, axis=1), np.std(cost_repair_NS, axis=1)
    cost_repair_C_mean, cost_repair_C_median, cost_repair_C_std =\
        np.mean(cost_repair_C, axis=1), np.median(cost_repair_C, axis=1), np.std(cost_repair_C, axis=1)
    cost_repair_D_mean, cost_repair_D_median, cost_repair_D_std =\
        np.mean(cost_repair_D, axis=1), np.median(cost_repair_D, axis=1), np.std(cost_repair_D, axis=1)
    cost_repair_ED_mean, cost_repair_ED_median, cost_repair_ED_std =\
        np.mean(cost_repair_ED, axis=1), np.median(cost_repair_ED, axis=1), np.std(cost_repair_ED, axis=1)
    cost_repair_A_mean, cost_repair_A_median, cost_repair_A_std =\
        np.mean(cost_repair_A, axis=1), np.median(cost_repair_A, axis=1), np.std(cost_repair_A, axis=1)
    cost_repair_L_mean, cost_repair_L_median, cost_repair_L_std =\
        np.mean(cost_repair_L, axis=1), np.median(cost_repair_L, axis=1), np.std(cost_repair_L, axis=1)
    cost_repair_LB_mean, cost_repair_LB_median, cost_repair_LB_std =\
        np.mean(cost_repair_LB, axis=1), np.median(cost_repair_LB, axis=1), np.std(cost_repair_LB, axis=1)
    cost_repair_V_mean, cost_repair_V_median, cost_repair_V_std =\
        np.mean(cost_repair_V, axis=1), np.median(cost_repair_V, axis=1), np.std(cost_repair_V, axis=1)
    cost_repair_NS_S_mean, cost_repair_NS_S_median, cost_repair_NS_S_std =\
        np.mean(cost_repair_NS_S, axis=1), np.median(cost_repair_NS_S, axis=1), np.std(cost_repair_NS_S, axis=1)
    n_Sa, n_MC = cost_total.shape
    
    get_log10_harzard_curve = interp1d(np.log10(hazard_curve[:, 0]), np.log10(hazard_curve[:, 1]),
                                       kind='cubic', fill_value='extrapolate', bounds_error=False)
    x_hazard = np.linspace(hazard_curve[0, 0], hazard_curve[-1, 0], 1000)  # 仅用于画图
    y_hazard = pow(10, get_log10_harzard_curve(np.log10(x_hazard)))
    log10_HSa = get_log10_harzard_curve(np.log10(Sa_ls))  # 灾害曲线的对数
    HSa = np.power(10, log10_HSa)  # 灾害曲线纵坐标
    diff_HSa = np.append(0, np.diff(HSa))  # 灾害曲线的差分
    
    # 年度经济损失(损伤-强度曲线与地震灾害曲线的乘积积分)
    EAL_total = float(np.sum(cost_total_mean * np.abs(diff_HSa)))
    EAL_clps = float(np.sum(cost_clps_mean * np.abs(diff_HSa)))
    EAL_dm = float(np.sum(cost_dm_mean * np.abs(diff_HSa)))
    EAL_repair = float(np.sum(cost_repair_mean * np.abs(diff_HSa)))
    EAL_repair_S = float(np.sum(cost_repair_S_mean * np.abs(diff_HSa)))
    EAL_repair_NS = float(np.sum(cost_repair_NS_mean * np.abs(diff_HSa)))
    EAL_repair_C = float(np.sum(cost_repair_C_mean * np.abs(diff_HSa)))
    EAL_repair_D = float(np.sum(cost_repair_D_mean * np.abs(diff_HSa)))
    EAL_repair_ED = float(np.sum(cost_repair_ED_mean * np.abs(diff_HSa)))
    EAL_repair_A = float(np.sum(cost_repair_A_mean * np.abs(diff_HSa)))
    EAL_repair_L = float(np.sum(cost_repair_L_mean * np.abs(diff_HSa)))
    EAL_repair_LB = float(np.sum(cost_repair_LB_mean * np.abs(diff_HSa)))
    EAL_repair_V = float(np.sum(cost_repair_V_mean * np.abs(diff_HSa)))
    EAL_repair_NS_S = float(np.sum(cost_repair_NS_S_mean * np.abs(diff_HSa)))
    
    # 年度超越概率-经济损失曲线
    # P (L > l | IM = Sa) = ∫ P (L > l | IM = Sa) * d λ(e)
    loss_ls = np.arange(0.01, 1, 0.01)
    annual_rate_total = np.zeros_like(loss_ls)
    annual_rate_clps = np.zeros_like(loss_ls)
    annual_rate_dm = np.zeros_like(loss_ls)
    annual_rate_rp = np.zeros_like(loss_ls)
    annual_rate_rp_S = np.zeros_like(loss_ls)
    annual_rate_rp_NS = np.zeros_like(loss_ls)
    annual_rate_rp_C = np.zeros_like(loss_ls)
    annual_rate_rp_D = np.zeros_like(loss_ls)
    annual_rate_rp_ED = np.zeros_like(loss_ls)
    annual_rate_rp_A = np.zeros_like(loss_ls)
    annual_rate_rp_L = np.zeros_like(loss_ls)
    annual_rate_rp_LB = np.zeros_like(loss_ls)
    annual_rate_rp_V = np.zeros_like(loss_ls)
    annual_rate_rp_NS_S = np.zeros_like(loss_ls)
    for idx_loss, loss in enumerate(loss_ls):
        # 建立地震强度Sa下的经济损失概率分布 P(L > l | IM = Sa)
        for idx_Sa, Sa in enumerate(Sa_ls):
            # 统计 cost_total[idx_Sa] 中大于等于 loss 的比例
            # 总
            P = len(cost_total[idx_Sa][cost_total[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_total[idx_loss] += P * d_lamda
            # 倒塌
            P = len(cost_clps[idx_Sa][cost_clps[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_clps[idx_loss] += P * d_lamda
            # 拆除
            P = len(cost_dm[idx_Sa][cost_dm[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_dm[idx_loss] += P * d_lamda
            # 总修复
            P = len(cost_repair[idx_Sa][cost_repair[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp[idx_loss] += P * d_lamda
            # 修复-结构构件(S)
            P = len(cost_repair_S[idx_Sa][cost_repair_S[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_S[idx_loss] += P * d_lamda
            # 修复-非结构构件(NS)
            P = len(cost_repair_NS[idx_Sa][cost_repair_NS[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_NS[idx_loss] += P * d_lamda
            # 修复-内容物(C)
            P = len(cost_repair_C[idx_Sa][cost_repair_C[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_C[idx_loss] += P * d_lamda
            # 修复-位移角(D)
            P = len(cost_repair_D[idx_Sa][cost_repair_D[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_D[idx_loss] += P * d_lamda
            # 修复-有效位移角(ED)
            P = len(cost_repair_ED[idx_Sa][cost_repair_ED[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_ED[idx_loss] += P * d_lamda
            # 修复-加速度(A)
            P = len(cost_repair_A[idx_Sa][cost_repair_A[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_A[idx_loss] += P * d_lamda
            # 修复-L
            P = len(cost_repair_L[idx_Sa][cost_repair_L[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_L[idx_loss] += P * d_lamda
            # 修复-LB
            P = len(cost_repair_LB[idx_Sa][cost_repair_LB[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_LB[idx_loss] += P * d_lamda
            # 修复-V
            P = len(cost_repair_V[idx_Sa][cost_repair_V[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_V[idx_loss] += P * d_lamda
            # 修复-位移型非结构构件(NS_S)
            P = len(cost_repair_NS_S[idx_Sa][cost_repair_NS_S[idx_Sa] > loss]) / n_MC
            d_lamda = np.abs(diff_HSa[idx_Sa])
            annual_rate_rp_NS_S[idx_loss] += P * d_lamda

    data = {
        'EAL_total': EAL_total,
        'EAL_collapse': EAL_clps,
        'EAL_demolishment': EAL_dm,
        'EAL_repair': EAL_repair,
        'EAL_repair_S': EAL_repair_S,
        'EAL_repair_NS': EAL_repair_NS,
        'EAL_repair_C': EAL_repair_C,
        'EAL_repair_D': EAL_repair_D,
        'EAL_repair_ED': EAL_repair_ED,
        'EAL_repair_A': EAL_repair_A,
        'EAL_repair_L': EAL_repair_L,
        'EAL_repair_LB': EAL_repair_LB,
        'EAL_repair_V': EAL_repair_V,
        'EAL_repair_NS_S': EAL_repair_NS_S
    }
    
    # 年度经济损失(超越概率-损失曲线在定义域上的积分)
    d_loss = np.diff(loss_ls)
    p1 = annual_rate_total[:-1]
    p2 = annual_rate_total[1:]
    EAL_total2 = float(np.sum((p1 + p2) / 2 * d_loss))
    print(f'Expected annual total loss ratio (= ∫0-∞ P(L > l)dL): {EAL_total2}')

    json.dump(data, open(output_dir / 'Expected annual loss ratio.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    np.savetxt(root / 'Hazard_USGS.txt', hazard_curve)
    np.savetxt(root / 'Hazard_fitting.txt', np.column_stack((x_hazard, y_hazard)))
    np.save(output_dir / 'annual_rate_total.npy', np.column_stack((loss_ls, annual_rate_total)))
    np.save(output_dir / 'annual_rate_clps.npy', np.column_stack((loss_ls, annual_rate_clps)))
    np.save(output_dir / 'annual_rate_dm.npy', np.column_stack((loss_ls, annual_rate_dm)))
    np.save(output_dir / 'annual_rate_rp.npy', np.column_stack((loss_ls, annual_rate_rp)))
    np.save(output_dir / 'annual_rate_rp_S.npy', np.column_stack((loss_ls, annual_rate_rp_S)))
    np.save(output_dir / 'annual_rate_rp_NS.npy', np.column_stack((loss_ls, annual_rate_rp_NS)))
    np.save(output_dir / 'annual_rate_rp_C.npy', np.column_stack((loss_ls, annual_rate_rp_C)))
    np.save(output_dir / 'annual_rate_rp_D.npy', np.column_stack((loss_ls, annual_rate_rp_D)))
    np.save(output_dir / 'annual_rate_rp_ED.npy', np.column_stack((loss_ls, annual_rate_rp_ED)))
    np.save(output_dir / 'annual_rate_rp_A.npy', np.column_stack((loss_ls, annual_rate_rp_A)))
    np.save(output_dir / 'annual_rate_rp_L.npy', np.column_stack((loss_ls, annual_rate_rp_L)))
    np.save(output_dir / 'annual_rate_rp_LB.npy', np.column_stack((loss_ls, annual_rate_rp_LB)))
    np.save(output_dir / 'annual_rate_rp_V.npy', np.column_stack((loss_ls, annual_rate_rp_V)))
    np.save(output_dir / 'annual_rate_rp_NS_S.npy', np.column_stack((loss_ls, annual_rate_rp_NS_S)))
    np.savetxt(output_dir / 'EAL_total.txt', np.array([EAL_total2]))
