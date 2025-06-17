import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from threading import Thread
from typing import Tuple
from .building import Building
from config.config import EDP_TYPING, LOGGER


def intensity_based_loss(
    n: int,
    Sa_ls: np.ndarray,
    building: Building,
    is_random: bool = True,
    random_seed: int = None,
    parallel: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """基于地震动强度计算建筑的直接经济损失 (支持并行+进度监控)"""

    cost_mat_clps = np.zeros((len(Sa_ls), n))
    cost_mat_dm = np.zeros((len(Sa_ls), n))
    cost_mat_repair = np.zeros((len(Sa_ls), n))

    if parallel <= 1:
        # 串行
        for idx_MC in range(n):
            idx_MC, cost_clps, cost_dm, cost_repair = _worker_wrapper(
                idx_MC, Sa_ls, building, is_random, random_seed, None)
            cost_mat_clps[:, idx_MC] = cost_clps
            cost_mat_dm[:, idx_MC] = cost_dm
            cost_mat_repair[:, idx_MC] = cost_repair
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
            for idx_MC, cost_clps, cost_dm, cost_repair in results:
                cost_mat_clps[:, idx_MC] = cost_clps
                cost_mat_dm[:, idx_MC] = cost_dm
                cost_mat_repair[:, idx_MC] = cost_repair
        monitor_thread.join()

    cost_total = cost_mat_clps + cost_mat_dm + cost_mat_repair
    cost_total /= building.replacement_cost
    cost_mat_clps /= building.replacement_cost
    cost_mat_dm /= building.replacement_cost
    cost_mat_repair /= building.replacement_cost

    return cost_total, cost_mat_clps, cost_mat_dm, cost_mat_repair

def _worker_wrapper(
        idx_MC: int,
        Sa_ls: np.ndarray,
        building: Building,
        is_random: bool,
        random_seed: int | None,
        queue
    ):
    try:
        if random_seed is not None:
            np.random.seed(random_seed + idx_MC)
        cost_clps, cost_dm, cost_repair = np.zeros_like(Sa_ls), np.zeros_like(Sa_ls), np.zeros_like(Sa_ls)
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
                edp_type: EDP_TYPING = comp.damage_states['edp_type']
                if edp_type == 'Story Drift Ratio':
                    edp = IDR[story - 1]
                elif edp_type == 'Acceleration':
                    if floor >= 2:
                        edp = PFA[floor - 2]
                    else:
                        edp = Sa
                else:
                    raise NotImplementedError('其他类型的EDP尚未实现')
                ds_flag = comp._simu_DS(edp, is_random)
                if ds_flag == 0:
                    cost_i = 0
                else:
                    cost_i = comp._get_cost(quantity, ds_flag, is_random)
                cost += cost_i
            cost_repair[idx_Sa] = cost
    except Exception as e:
        LOGGER.error(f"Error in Monte Carlo simulation {idx_MC}: {e}")
        print(e)
        raise e
    if queue is not None:
        queue.put(1)
    return idx_MC, cost_clps, cost_dm, cost_repair
