import json
from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from config.config import LOGGER


def visualize_IBL(IBL_folder: str | Path):
    """可视化基于地震动强度的直接经济损失"""
    IBL_folder = Path(IBL_folder)
    Sa = np.load(IBL_folder / 'Sa.npy')
    cost_total = np.load(IBL_folder / 'cost_total.npy') * 100
    cost_clps = np.load(IBL_folder / 'cost_collapse.npy') * 100
    cost_dm = np.load(IBL_folder / 'cost_demolishment.npy') * 100
    cost_repair = np.load(IBL_folder / 'cost_repair.npy') * 100
    cost_total_mean = np.mean(cost_total, axis=1)
    cost_clps_mean = np.mean(cost_clps, axis=1)
    cost_dm_mean = np.mean(cost_dm, axis=1)
    cost_repair_mean = np.mean(cost_repair, axis=1)
    cost_rapair_category: dict[str, np.ndarray] = {'S': None, 'NS': None, 'C': None}
    cost_repair_sensitivity: dict[str, np.ndarray] = {
        'D': None, 'A': None, 'ED': None, 'L': None, 'LB': None, 'V': None}
    for key in cost_rapair_category:
        cost_rapair_category[key] = np.load(IBL_folder / f'cost_repair_category_{key}.npy') * 100
    for key in cost_repair_sensitivity:
        cost_repair_sensitivity[key] = np.load(IBL_folder / f'cost_repair_sensitivity_{key}.npy') * 100
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
    # 不同类型(修复、拆除、倒塌)损失堆叠面积图
    plt.fill_between(Sa, 0, cost_repair_mean/cost_total_mean, color='skyblue', alpha=0.7,
                     label='Repair', edgecolor='black')
    plt.fill_between(Sa, cost_repair_mean/cost_total_mean, (cost_repair_mean+cost_dm_mean)/cost_total_mean,
                     color='lightgreen', alpha=0.7, label='Demolishment', edgecolor='black')
    plt.fill_between(Sa, (cost_repair_mean+cost_dm_mean)/cost_total_mean, 1, color='salmon', alpha=0.7,
                     label='Collapse', edgecolor='black')
    plt.xlim(np.min(Sa), np.max(Sa))
    plt.ylim(0, 1)
    plt.xlabel('Sa (g)')
    plt.ylabel('Proportion')
    plt.title('Disaggregation of economic loss')
    plt.legend()
    plt.subplot(223)
    # 不同构件类型(S, NS, C)损失堆叠面积图
    sum_curve = sum([np.mean(data, axis=1) for data in cost_rapair_category.values()])
    # 截取sum_curve直至首个元素等于0为止
    cond = np.where(sum_curve == 0)[0][0]
    sum_curve = sum_curve[:cond]
    bot_curve = np.zeros_like(Sa)[:cond]
    top_curve = np.zeros_like(Sa)[:cond]
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'pink']
    for i, (key, data) in enumerate(cost_rapair_category.items()):
        top_curve += np.mean(data, axis=1)[:cond] / sum_curve
        if i == 0:
            bot_curve = 0
        plt.fill_between(Sa[:cond], bot_curve, top_curve, color=colors[i], alpha=0.7,
                         label=key, edgecolor='black')
        bot_curve = top_curve.copy()
    plt.xlim(np.min(Sa[:cond]), np.max(Sa[:cond]))
    plt.ylim(0, 1)
    plt.xlabel('Sa (g)')
    plt.ylabel('Proportion')
    plt.title('Repair cost disaggregation by component type')
    plt.legend(loc='lower right')
    plt.subplot(224)
    # 不同构件的敏感性类型(D, ED, A, L, LB, V)损失堆叠面积图
    sum_curve = sum([np.mean(data, axis=1) for data in cost_repair_sensitivity.values()])
    # 截取sum_curve直至首个元素等于0为止
    cond = np.where(sum_curve == 0)[0][0]
    sum_curve = sum_curve[:cond]
    bot_curve = np.zeros_like(Sa)[:cond]
    top_curve = np.zeros_like(Sa)[:cond]
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'pink']
    for i, (key, data) in enumerate(cost_repair_sensitivity.items()):
        top_curve += np.mean(data, axis=1)[:cond] / sum_curve
        if i == 0:
            bot_curve = 0
        plt.fill_between(Sa[:cond], bot_curve, top_curve, color=colors[i], alpha=0.7,
                         label=key, edgecolor='black')
        bot_curve = top_curve.copy()
    plt.xlim(np.min(Sa[:cond]), np.max(Sa[:cond]))
    plt.ylim(0, 1)
    plt.xlabel('Sa (g)')
    plt.ylabel('Proportion')
    plt.title('Repair cost disaggregation by EDP sensitivity')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(IBL_folder / 'IBL.png', dpi=600)
    plt.show()

def visualize_TBL(TBL_folder: str | Path):
    """可视化基于时间的直接经济损失"""
    TBL_folder = Path(TBL_folder)
    data = json.load(open(TBL_folder / 'Expected annual loss ratio.json', 'r', encoding='utf-8'))
    EAL_clps = data['EAL_collapse']
    EAL_dm = data['EAL_demolishment']
    EAL_repair = data['EAL_repair']
    EAL_total = data['EAL_total']

    hazard_curve = np.load(TBL_folder / 'USGS_hazard.npy')
    x_hazard, y_hazard = np.load(TBL_folder / 'hazard_fitting.npy').T
    loss_ls, annual_rate = np.load(TBL_folder / 'annual_rate.npy').T

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
    plt.subplot(223)
    plt.bar(['Collapse', 'Demolishment', 'Repair', 'Total'],
            [EAL_clps*100, EAL_dm*100, EAL_repair*100, EAL_total*100])
    plt.ylabel('Loss Ratio (%)')
    plt.title('Expected Annual Loss Ratio')
    plt.subplot(224)
    plt.plot(loss_ls, annual_rate, label='Annual Loss Ratio')
    plt.xlabel('Loss Ratio')
    plt.ylabel('Annual probability of exceedance')
    plt.title('Annual Rate - Economic Loss Curve')
    plt.tight_layout()
    plt.savefig(TBL_folder / 'TBL.png', dpi=600)
    plt.show()