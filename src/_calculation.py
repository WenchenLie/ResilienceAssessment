from math import isclose
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.config import LOGGER


def get_EDP_fragility(
    Sa_ls: np.ndarray,
    edp_ls: np.ndarray,
    gm_num: int,
    IDA_data: list[np.ndarray]
):
    if isclose(edp_ls[0], 0):
        raise ValueError('EDP值不能以0开头')
    for idx_edp, edp in enumerate(edp_ls):
        IM_ls = np.zeros(gm_num)  # 给定EDP值下的IM值
        for idx_gm, curve in enumerate(IDA_data):
            x, y = curve[:, 0], curve[:, 1]  # 单条IDA曲线的横、纵坐标
            if not min(x) <= edp <= max(x):
                raise ValueError(f'EDP值({edp})超出IDA曲线范围({min(x)}, {max(x)})')
            yi = _get_y(x, y, edp)  # 给定EDP值下，IDA曲线对应的地震强度
            IM_ls[idx_gm] = yi




def _get_y(x: list, y: list, x0: float, error: bool=True) -> float:
    """获得竖线x=x0与给定曲线的交点纵坐标

    Args:
        x (list): 输入曲线的横坐标序列
        y (list): 输入曲线的纵坐标序列
        x0 (float): 竖直线x = x0
        error (boo, optional): 若x0超出范围，抛出异常or返回None

    Returns:
        float: 曲线与竖线交点纵坐标
    """
    # 获得x=x0与曲线的交点
    if x0 < min(x):
        if error:
            raise ValueError(f'【Error】x0 < min(x) ({x0} < {min(x)})')
        else:
            return None
    if x0 > max(x):
        if error:
            raise ValueError(f'【Error】x0 > max(x) ({x0} > {max(x)})')
        else:
            return None
    for i in range(len(x) - 1):
        if x[i] == x0:
            y0 = y[i]
            return y0
        elif x[i] < x0 <= x[i + 1]:
            k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            y0 = k * (x0 - x[i]) + y[i]
            return y0
    else:
        raise ValueError('【Error】未找到交点-2')


def exceedance_probability(self,
        EDP_type: str,
        DM_value: float,
        beta: list[float] | float=None,
        *args, **kwargs
    ):
    """计算所有损伤指标超越某值的概率

    注1: 
    -----
    通过该方法计算的易损性函数具有整体表达式，通常用于计算结构的倒塌易损性    
    FR(x) = P[D|IM=x] = DCF((ln(mD(x) - ln(mC)) / sqrt(beta^2))),  
    其中, `mD(x)`是中值, `mC`是能力值, `beta`是标准差 
    应注意的是，mD(x)不根据概率地震需求模型进行计算，而是直接使用IM的中值，
    此外，β可指定也可以为`None`，详见注2  

    注2:
    ------
    不确定性参数`beta`可填多个值，
    多个`beta`值将按下式进行叠加: `beta = sqrt(beta_1^2 + beta_2^2 + beta_3^2 + ...)`
    例如：  
    >>> beta = [0.2, 0.3, 0.4]
    表示beta = sqrt(0.2^2 + 0.3^2 + 0.4^2)
    也可只传入单个不确定值，例如：  
    >>> beta = 0.4  

    无论`beta`是否为`None`，程序都会按以下情况计算一次beta值：  
    beta = std(IM(D), ddof=1)  
    即指定需求D下对应的所有IM值的标准差  
    在输出结果时，会同时输出指定`beta`值(如果指定了的话)或使用默认计算的`beta`下的易损性曲线  

    注3:
    ------
    当`EDP_type`定义为`'IDR'`时，基于给定`beta`值计算的倒塌易损性曲线将用于倒塌评估方法`collapse_evaluation`，
    如果`beta`为`None`，则倒塌易损性曲线的计算使用默认计算的`beta`值

    Args:
        EDP_type (str): 工程需求参数名称
        DM_values (float): 损伤指标超越DM_value的概率
        beta (list[float] | float, optional): 指定确定性来绘制易损性曲线，默认None，即仅考虑计算得到的地震动记录不确定性
    """
    internal_call: bool = kwargs.get('internal_call', False)
    if not EDP_type in self.EDP_types:
        raise KeyError(f'尚未指定`{EDP_type}`类型，请在`__init__`方法的`EDP_types`参数中添加')
    # EDP_type: 损伤指标名称, DM_value: 损伤值
    # 以直线x=DM_value切割所有IDA曲线，获得交点
    IM_lines, DM_lines = self.IM_lines[EDP_type], self.DM_lines[EDP_type]  # 所有IDA曲线
    if DM_value < min([min(i) for i in DM_lines]):
        raise ValueError(f'`{EDP_type}`类型的`{DM_value}`值小于所有IDA曲线的最小DM值({min([max(i) for i in DM_lines])})')
    if DM_value > max([max(i) for i in DM_lines]):
        raise ValueError(f'`{EDP_type}`类型的`{DM_value}`值大于所有IDA曲线的最大DM值({max([max(i) for i in DM_lines])})')
    IM_points = []  # 交点对应的IM纵坐标列表
    for i in range(self.GM_N):
        IM_line, DM_line = IM_lines[i], DM_lines[i]  # 单调IDA曲线
        try:
            y = get_y(DM_line, IM_line, DM_value)
            IM_points.append(y)
        except ValueError:
            LOGGER.warning(f'`{EDP_type}`类型的`{DM_value}`值不在第{i+1}条IDA曲线的DM范围内({min(DM_line)}, {max(DM_line)})')
    if len(IM_points) < self.GM_N:
        LOGGER.error(f'将不进行`{EDP_type}`类型的超越概率统计')
        return
    exceed_mean, exceed_std, exceed_pct50 = np.mean(IM_points), np.std(IM_points, ddof=1), np.percentile(IM_points, 50)  # 均值、标准差、中位值
    exceed_x: list[float] = sorted(IM_points)  # 超越概率点横坐标(IM)
    exceed_y = np.array([i/self.GM_N for i in range(1, self.GM_N+1)])  # 超越概率点纵坐标(超越概率)
    # 确定β
    if beta is not None:
        try:
            iter(beta)
        except TypeError:
            beta = [beta]
        beta_total = 0
        for beta_i in beta:
            beta_total += beta_i ** 2
        beta_total = float(np.sqrt(beta_total))
    else:
        beta_total = None
    # 拟合cdf曲线
    epsilon = 1e-4
    z = norm.ppf(np.clip(exceed_y, epsilon, 1 - epsilon))
    lnIM = np.log(exceed_x)
    A = np.vstack([z, np.ones_like(z)]).T  # 线性最小二乘法拟合
    beta_calc, ln_theta = np.linalg.lstsq(A, lnIM, rcond=None)[0]  # 对数标准差，对数中值
    theta = np.exp(ln_theta)
    # theta = np.median(exceed_x)  # 取中值
    # beta_calc = np.std(np.log(exceed_x), ddof=1)  # 对数标准差
    IM1, IM2 = 0.001, max(exceed_x) * 1.2  # 坐标范围
    exceed_x_fit = np.linspace(IM1, IM2, 1001)  # 超越概率曲线横坐标(IM)
    exceed_y_fit = norm.cdf(np.log(exceed_x_fit / theta) / beta_calc, 0, 1)  # 超越概率曲线横坐标(超越概率)
    if beta_total is not None:
        beta_fixed = beta_total  # 采用传入的总不确定性(beta_TOT)
        if not internal_call:
            LOGGER.success(f'为`{EDP_type}`类型指定了体系总不确定性: beta_total = {beta_fixed}')
        self.DM_has_fixed_beta[EDP_type] = beta_fixed
    else:
        beta_fixed = beta_calc
    exceed_y_fixedBeta = norm.cdf(np.log(exceed_x_fit / theta) / beta_fixed, 0, 1)
    text = ''  # 统计特征的文本
    text += f'`{EDP_type}`超越{DM_value}的概率特征：\n'
    text += f'均值：{exceed_mean:.6f}\n'
    text += f'标准差：{exceed_std:.6f}\n'
    text += f'中位值：{exceed_pct50:.6f}'
    if not internal_call:
        print(text)
        self.exceed_mean[EDP_type] = exceed_mean
        self.exceed_std[EDP_type] = exceed_std
        self.exceed_pct50[EDP_type] = exceed_pct50
        self.exceed_x[EDP_type] = exceed_x
        self.exceed_y[EDP_type] = exceed_y
        self.exceed_x_fit[EDP_type] = exceed_x_fit
        self.exceed_y_fit[EDP_type] = exceed_y_fit
        self.exceed_y_fixedBeta[EDP_type] = exceed_y_fixedBeta
        self.info[EDP_type] += text
        self.DM_values[EDP_type] = DM_value
        # 画图: 超越概率曲线
        fig: Figure = self.all_figures_1[EDP_type]
        axes = fig.get_axes()
        ax: Axes = axes[3]
        ax.set_title(f'Exceedance probability of {EDP_type}>{DM_value}')
        ax.set_ylabel(f'P({EDP_type}>{DM_value})')
        ax.set_xlabel('IM')
        if EDP_type in self.exceed_x_fit.keys():
            ax.plot(exceed_x_fit, exceed_y_fit, label=f'Computed beta = {beta_calc:.3f}')
            ax.plot(exceed_x, exceed_y, 'o', color='red')
            if EDP_type in self.DM_has_fixed_beta.keys():
                ax.plot(exceed_x_fit, exceed_y_fixedBeta, label=f'Fixed beta = {beta_fixed:.3f}')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.legend()
        LOGGER.success('已计算超越概率曲线')
    else:
        # 内部调用的情况，用于进行倒塌易损性评估
        # 返回倒塌强度中值，实际标准差，倒塌易损性横坐标，倒塌易损性纵坐标，固定总不确定性下的易损性纵坐标，所有倒塌强度点，倒塌易损性曲线
        N_lines = max(len(exceed_x), len(exceed_x_fit), len(exceed_x_fit))
        df = pd.DataFrame(None)
        df['Sa_scatters'] = pd.Series(exceed_x).reindex(range(N_lines))  # 倒塌易损性曲线-散点
        df['IM_scatters'] = pd.Series(exceed_y).reindex(range(N_lines))
        df['Sa_fixedBeta'] = pd.Series(exceed_x_fit).reindex(range(N_lines))  # 倒塌易损性曲线-固定总不确定性
        df['IM_fixedBeta'] = pd.Series(exceed_y_fixedBeta).reindex(range(N_lines))
        df['Sa_fit'] = pd.Series(exceed_x_fit).reindex(range(N_lines))  # 倒塌易损性曲线-按计算得到的对数标准差
        df['IM_fit'] = pd.Series(exceed_y_fit).reindex(range(N_lines))
        return theta, beta_calc, exceed_x_fit, exceed_y_fit, exceed_y_fixedBeta, exceed_x, df
