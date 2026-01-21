import os
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.config import LOGGER


def post_processing(root: str | Path, plot: bool = True):
    root = Path(root)
    output_dir = root / 'Post-processing'
    if not output_dir.exists():
        os.makedirs(output_dir)
    _visualize_IBL(root, output_dir, plot)
    _visualize_TBL(root, output_dir, plot)
    _visualize_repair_time(root, output_dir, plot)
    _visualize_casualties(root, output_dir, plot)


def _visualize_IBL(root: Path, output_dir: Path, plot: bool):
    """可视化基于地震动强度的直接经济损失"""
    Sa = np.loadtxt(root / 'Sa.txt')
    cost_total = np.load(root / 'Repair cost/cost_total.npy') * 100
    cost_clps = np.load(root / 'Repair cost/cost_collapse.npy') * 100
    cost_dm = np.load(root / 'Repair cost/cost_demolishment.npy') * 100
    cost_repair = np.load(root / 'Repair cost/cost_repair.npy') * 100
    cost_total_mean = np.mean(cost_total, axis=1)  # NOTE: 倒塌和拆除是二元数据，无法取中值
    cost_clps_mean = np.mean(cost_clps, axis=1)
    cost_dm_mean = np.mean(cost_dm, axis=1)
    cost_repair_mean = np.mean(cost_repair, axis=1)
    cost_rapair_category: dict[str, np.ndarray] = {'S': None, 'NS': None, 'C': None}
    cost_repair_sensitivity: dict[str, np.ndarray] = {
        'D': None, 'A': None, 'ED': None, 'L': None, 'LB': None, 'V': None}
    for key in cost_rapair_category:
        cost_rapair_category[key] = np.load(root / f'Repair cost/cost_repair_category_{key}.npy') * 100
    for key in cost_repair_sensitivity:
        cost_repair_sensitivity[key] = np.load(root / f'Repair cost/cost_repair_sensitivity_{key}.npy') * 100
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.plot(Sa, cost_clps_mean, label='Collapse')
    plt.legend()
    plt.plot(Sa, cost_dm_mean, label='Demolition')
    plt.legend()
    plt.plot(Sa, cost_repair_mean, label='Repair')
    plt.legend()
    plt.plot(Sa, cost_total_mean, label='Total')
    plt.legend()
    plt.xlabel('Sa (g)')
    plt.ylabel('Economic Loss')
    plt.title('Economic Loss - Sa Curve')
    plt.subplot(222)
    df = pd.DataFrame(np.column_stack((Sa, cost_clps_mean, cost_dm_mean, cost_repair_mean, cost_total_mean)),
                      columns=['Sa', 'Collapse', 'Demolishment', 'Repair', 'Total'])
    df.to_csv(output_dir / 'Intensity-based loss.csv', index=False)
    # 不同类型(修复、拆除、倒塌)损失堆叠面积图
    curve1 = cost_repair_mean / cost_total_mean
    curve2 = (cost_repair_mean + cost_dm_mean) / cost_total_mean
    plt.fill_between(Sa, 0, curve1, color='skyblue', alpha=0.7, label='Repair', edgecolor='black')
    plt.fill_between(Sa, curve1, curve2, color='lightgreen', alpha=0.7, label='Demolishment', edgecolor='black')
    plt.fill_between(Sa, curve2, 1, color='salmon', alpha=0.7, label='Collapse', edgecolor='black')
    plt.xlim(np.min(Sa), np.max(Sa))
    plt.ylim(0, 1)
    plt.xlabel('Sa (g)')
    plt.ylabel('Proportion')
    plt.title('Disaggregation of economic loss')
    plt.legend()
    df = pd.DataFrame(np.column_stack((Sa, np.zeros_like(Sa), curve1, curve2, np.ones_like(Sa))), columns=['Sa', '0', 'Repair', 'Demolishment', 'Collapse'])
    df.to_csv(output_dir / 'Repair cost disaggregation-1.csv', index=False)
    plt.subplot(223)
    # 不同构件类型(S, NS, C)损失堆叠面积图
    sum_curve = sum([np.mean(data, axis=1) for data in cost_rapair_category.values()])
    # 截取sum_curve直至首个元素等于0为止
    try:
        cond = np.where(sum_curve[3:] == 0)[0][0]
    except IndexError:
        cond = len(sum_curve)
    sum_curve = sum_curve[:cond]
    bot_curve = np.zeros_like(Sa)[:cond]
    top_curve = np.zeros_like(Sa)[:cond]
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'pink']
    curves = []
    for i, (key, data) in enumerate(cost_rapair_category.items()):
        top_curve += np.mean(data, axis=1)[:cond] / sum_curve
        if i == 0:
            bot_curve = 0
        plt.fill_between(Sa[:cond], bot_curve, top_curve, color=colors[i], alpha=0.7,
                         label=key, edgecolor='black')
        curves.append(top_curve.copy())
        bot_curve = top_curve.copy()
    if cond != 0:
        plt.xlim(np.min(Sa[:cond]), np.max(Sa[:cond]))
    plt.ylim(0, 1)
    plt.xlabel('Sa (g)')
    plt.ylabel('Proportion')
    plt.title('Repair cost disaggregation by component type')
    plt.legend(loc='lower right')
    df = pd.DataFrame(np.column_stack((Sa[:cond], np.zeros_like(Sa[:cond]), *curves)), columns=['Sa', '0', *list(cost_rapair_category.keys())])
    df.to_csv(output_dir / 'Repair cost disaggregation-2.csv', index=False)
    plt.subplot(224)
    # 不同构件的敏感性类型(D, ED, A, L, LB, V)损失堆叠面积图
    sum_curve = sum([np.mean(data, axis=1) for data in cost_repair_sensitivity.values()])
    # 截取sum_curve直至首个元素等于0为止
    try:
        cond = np.where(sum_curve[3:] == 0)[0][0]
    except IndexError:
        cond = len(sum_curve)
    sum_curve = sum_curve[:cond]
    bot_curve = np.zeros_like(Sa)[:cond]
    top_curve = np.zeros_like(Sa)[:cond]
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'pink']
    curves = []
    for i, (key, data) in enumerate(cost_repair_sensitivity.items()):
        top_curve += np.mean(data, axis=1)[:cond] / sum_curve
        if i == 0:
            bot_curve = 0
        plt.fill_between(Sa[:cond], bot_curve, top_curve, color=colors[i], alpha=0.7,
                         label=key, edgecolor='black')
        curves.append(top_curve.copy())
        bot_curve = top_curve.copy()
    df = pd.DataFrame(np.column_stack((Sa[:cond], np.zeros_like(Sa[:cond]), *curves)), columns=['Sa', '0', *list(cost_repair_sensitivity.keys())])
    df.to_csv(output_dir / 'Repair cost disaggregation-3.csv', index=False)
    if cond != 0:
        plt.xlim(np.min(Sa[:cond]), np.max(Sa[:cond]))
    plt.ylim(0, 1)
    plt.xlabel('Sa (g)')
    plt.ylabel('Proportion')
    plt.title('Repair cost disaggregation by EDP sensitivity')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'Intensity-based loss.png', dpi=600)
    if plot:
        plt.show()
    else:
        plt.close()


def _visualize_TBL(root: Path, output_dir: Path, plot: bool):
    """可视化基于时间的直接经济损失"""
    data = json.load(open(root / 'Repair cost/Expected annual loss ratio.json', 'r', encoding='utf-8'))
    EAL_clps = data['EAL_collapse']
    EAL_dm = data['EAL_demolishment']
    EAL_repair = data['EAL_repair']
    EAL_total = data['EAL_total']

    hazard_curve = np.loadtxt(root / 'Hazard_USGS.txt')
    x_hazard, y_hazard = np.loadtxt(root / 'Hazard_fitting.txt').T
    loss_ls, annual_rate_total = np.load(root / 'Repair cost/annual_rate_total.npy').T
    loss_ls, annual_rate_clps = np.load(root / 'Repair cost/annual_rate_clps.npy').T
    loss_ls, annual_rate_dm = np.load(root / 'Repair cost/annual_rate_dm.npy').T
    loss_ls, annual_rate_rp = np.load(root / 'Repair cost/annual_rate_rp.npy').T
    loss_ls, annual_rate_rp_S = np.load(root / 'Repair cost/annual_rate_rp_S.npy').T
    loss_ls, annual_rate_rp_NS = np.load(root / 'Repair cost/annual_rate_rp_NS.npy').T
    loss_ls, annual_rate_rp_C = np.load(root / 'Repair cost/annual_rate_rp_C.npy').T
    loss_ls, annual_rate_rp_D = np.load(root / 'Repair cost/annual_rate_rp_D.npy').T
    loss_ls, annual_rate_rp_ED = np.load(root / 'Repair cost/annual_rate_rp_ED.npy').T
    loss_ls, annual_rate_rp_A = np.load(root / 'Repair cost/annual_rate_rp_A.npy').T
    loss_ls, annual_rate_rp_L = np.load(root / 'Repair cost/annual_rate_rp_L.npy').T
    loss_ls, annual_rate_rp_LB = np.load(root / 'Repair cost/annual_rate_rp_LB.npy').T
    loss_ls, annual_rate_rp_V = np.load(root / 'Repair cost/annual_rate_rp_V.npy').T
    loss_ls, annual_rate_rp_NS_S = np.load(root / 'Repair cost/annual_rate_rp_NS_S.npy').T

    LOGGER.info(f'Expected annual loss ratio (total): {EAL_total:.2%}')
    LOGGER.info(f'Expected annual loss ratio (collapse): {EAL_clps:.2%}')
    LOGGER.info(f'Expected annual loss ratio (demolishment): {EAL_dm:.2%}')
    LOGGER.info(f'Expected annual loss ratio (repair): {EAL_repair:.2%}')
    
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.loglog(hazard_curve[:, 0], hazard_curve[:, 1], '-o', label='USGS data')
    plt.loglog(x_hazard, y_hazard, label='Cubic fitting')
    plt.grid(True)
    plt.xlabel('Sa')
    plt.ylabel(f'MAF of Sa')
    plt.title('Hazard Curve')
    plt.legend()
    plt.subplot(222)
    plt.pie([EAL_clps, EAL_dm, EAL_repair], labels=['Collapse', 'Demolishment', 'Repair'], autopct='%1.2f%%')
    plt.title(f'Disaggregation of aunual economic loss')
    plt.legend(loc='upper left')
    df = pd.DataFrame(None, columns=['Type', 'EAL'])
    df['Type'] = ['Collapse', 'Demolishment', 'Repair']
    df['EAL'] = [EAL_clps, EAL_dm, EAL_repair]
    df.to_csv(output_dir / 'EAL dissaggregation.csv', index=False)
    plt.subplot(223)
    plt.bar(['Collapse', 'Demolishment', 'Repair', 'Total'],
            [EAL_clps*100, EAL_dm*100, EAL_repair*100, EAL_total*100])
    plt.ylabel('Loss Ratio (%)')
    plt.title('Expected Annual Loss Ratio')
    df = pd.DataFrame(None, columns=['Type', 'EAL'])
    df['Type'] = ['Collapse', 'Demolishment', 'Repair', 'Total']
    df['EAL'] = [EAL_clps*100, EAL_dm*100, EAL_repair*100, EAL_total*100]
    df.to_csv(output_dir / 'EAL ratio.csv', index=False)
    plt.subplot(224)
    plt.plot(loss_ls, annual_rate_total, label='Annual Loss Ratio (total)')
    plt.xlabel('Loss Ratio')
    plt.ylabel('Annual probability of exceedance')
    plt.title('Annual Rate - Economic Loss Curve')
    plt.tight_layout()
    plt.savefig(output_dir / 'Time-based loss.png', dpi=600)
    arr = np.column_stack((loss_ls,
                           annual_rate_total,
                           annual_rate_clps,
                           annual_rate_dm,
                           annual_rate_rp,
                           annual_rate_rp_S,
                           annual_rate_rp_NS,
                           annual_rate_rp_C,
                           annual_rate_rp_D,
                           annual_rate_rp_ED,
                           annual_rate_rp_A,
                           annual_rate_rp_L,
                           annual_rate_rp_LB,
                           annual_rate_rp_V,
                           annual_rate_rp_NS_S
                           ))
    cols = ['Loss Ratio', 'Total', 'Collapse', 'Demolition', 'Repair', 'Repair_S', 'Repair_NS', 'Repair_C',
            'Repair_D', 'Repair_ED', 'Repair_A', 'Repair_L', 'Repair_LB', 'Repair_V', 'Repair_NS_S']
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(output_dir / 'Time-based loss.csv', index=False)
    if plot:
        plt.show()
    else:
        plt.close()
    

def _visualize_repair_time(root: Path, output_dir: Path, plot: bool):
    Sa = np.loadtxt(root / 'Sa.txt')
    repair_time: np.ndarray = np.load(root / 'Repair time/repair_time.npy')
    repair_time_mean = np.mean(repair_time, axis=1)
    plt.plot(Sa, repair_time_mean)
    plt.xlabel('Sa (g)')
    plt.ylabel('Time (days)')
    plt.title('Repair Time')
    plt.tight_layout()
    plt.savefig(output_dir / 'Repair time.png', dpi=600)
    if plot:
        plt.show()
    else:
        plt.close()


def _visualize_casualties(root: Path, output_dir: Path, plot: bool):
    Sa = np.loadtxt(root / 'Sa.txt')
    death_rate: np.ndarray = np.load(root / 'Casualties/Death_rate.npy') * 100
    injury_rate: np.ndarray = np.load(root / 'Casualties/Injury_rate.npy') * 100
    death_rate = np.mean(death_rate, axis=1)
    injury_rate = np.mean(injury_rate, axis=1)
    plt.plot(Sa, death_rate, label='Death rate')
    plt.plot(Sa, injury_rate, label='Injury rate')
    plt.xlabel('Sa (g)')
    plt.ylabel('Rate (%)')
    plt.title('Casualty Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'Casualty rate.png', dpi=600)
    if plot:
        plt.show()
    else:
        plt.close()
