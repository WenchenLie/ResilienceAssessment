from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def visualize_IBL(IBL_folder: str | Path):
    """可视化基于地震动强度的直接经济损失"""
    IBL_folder = Path(IBL_folder)
    Sa = np.load(IBL_folder / 'Sa.npy')
    cost_total = np.load(IBL_folder / 'cost_total.npy') * 100
    cost_clps = np.load(IBL_folder / 'cost_collapse.npy') * 100
    cost_dm = np.load(IBL_folder / 'cost_demolishment.npy') * 100
    cost_repair = np.load(IBL_folder / 'cost_repair.npy') * 100
    plt.plot(Sa, np.mean(cost_clps, axis=1), label='Collapse')
    plt.legend()
    plt.plot(Sa, np.mean(cost_dm, axis=1), label='Demolition')
    plt.legend()
    plt.plot(Sa, np.mean(cost_repair, axis=1), label='Repair')
    plt.legend()
    plt.plot(Sa, np.mean(cost_total, axis=1), label='Total')
    plt.legend()
    plt.xlabel('Sa (g)')
    plt.ylabel('Economic Loss')
    plt.tight_layout()
    plt.show()

