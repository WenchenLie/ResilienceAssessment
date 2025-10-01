import json
from pathlib import Path
from math import isclose
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as px
# from scipy.stats import norm
from openpyxl.worksheet.worksheet import Worksheet
from .compenent import Component
from .unit_convertor import to_foot
from ._calculation import _cdf, _normal, _lognormal
from config.config import UNITS_TYPING, LOGGER,\
    VECTOR, OCCUPANCIES_TYPING


class Building:
    root = Path(__file__).parent.parent
    
    def __init__(self,
            name: str,
            Nstory: int,
            size: tuple[float, float],
            heights: list[float],
            unit: UNITS_TYPING,
            replacement_cost: float,
            replacement_time: float,
            occupancy: OCCUPANCIES_TYPING,
            collapse_fatality_rate: float=0.01,
            collapse_fatality_COV: float=0.5,
            collapse_injury_rate: float=0.01,
            collapse_injury_COV: float=0.5,
        ):
        """定义一栋建筑，并设置基本参数

        Args:
            name (str): 名称
            Nstory (int): 楼层数
            size (tuple[float, float]): 平面尺寸
            heights (list[float]): 层高列表
            unit (UNITS_TYPING): 尺寸和层高的单位
            replacement_cost (float): 重建成本
            replacement_time (float): 重建时间
            occupancy (OCCUPANCIES_TYPING): 建筑使用功能
            collapse_fatality_rate (float, optional): 倒塌死亡率，默认100%
            collapse_fatality_COV (float, optional): 倒塌死亡率的协方差，默认0.5
            collapse_injury_rate (float, optional): 倒塌受伤率，默认100%
            collapse_injury_COV (float, optional): 倒塌受伤率的协方差，默认0.5
        
        Notes:
        ------
        所有长度量纲的单位都将转换为英尺(foot)
        """
        self.name = name
        self.Nstory = Nstory
        self.size = to_foot(size, unit)
        self.floor_area = self.size[0] * self.size[1]
        self.heights = to_foot(heights, unit)
        self.replacement_cost = replacement_cost
        self.replacement_time = replacement_time
        self.occupancy = occupancy
        self.collapse_fatality_rate = collapse_fatality_rate
        self.collapse_fatality_COV = collapse_fatality_COV
        self.collapse_injury_rate = collapse_injury_rate
        self.collapse_injury_COV = collapse_injury_COV
        self.components: list[tuple[Component, float, int, int]] = []
        self.account_for_clps: bool = False  # 是否考虑倒塌
        self.account_for_dm: bool = False  # 是否考虑残余变形过大导致的拆除
        self._init_population()
        LOGGER.success(f'Building "{self.name}" is created successfully.')

    def _init_population(self):
        """根据建筑使用功能读取人口模型"""
        pop_model: dict = json.load(open(self.root / f'data/population models/{self.occupancy}.json', 'r'))
        self.pop_num: float = pop_model['Peak number per 1000sf']
        self.pop_beta: float = pop_model['Dispersion']
        self.pop_day: dict[str, list[float, float]] = pop_model['Day']
        self.pop_month: dict[str, list[float, float]] = pop_model['Month']

    def add_component(self,
            component: Component,
            quantity: float,
            story: int = None,
            floor: int = None
        ):
        """添加建筑物的构件(包括所有结构构件、非结构构件和建筑内容)

        Args:
            component (Component): Component对象
            quantity (float): 数量
            story (int, optional): 楼层编号
            floor (int, optional): 楼面编号
        
        Note:
        -----
        * `story`和`floor`不得同时为`None`
        """
        if story is None and floor is None:
            raise ValueError('Either `story` or `floor` should be provided.')
        self.components.append((component, quantity, story, floor))
    
    def add_IDAdata(self,
            gm_num: int,
            file_IDR: str | Path,
            file_RIDR: str | Path,
            file_PFA: str | Path,
            file_PFV: str | Path = None,
            IDR_factor: float = 1.0,
            RIDR_factor: float = 1.0,
            PFA_factor: float = 1.0,
            PFV_factor: float = 1.0,
            IDR_range: tuple[float, float] = None,
            RIDR_range: tuple[float, float] = None,
            PFA_range: tuple[float, float] = None,
            PFV_range: tuple[float, float] = None,
        ):
        """导入IDA结果的csv文件，格式可参考`template_*.csv`

        Args:
            gm_num (int): 地震动数量
            file_IDR (str | Path): 层间位移角(rad)
            file_RIDR (str | Path): 残余层间位移角(rad)
            file_PFA (str | Path): 楼层绝对加速度(g)
            file_PFV (str | Path, optional): 楼层绝对速度(in/s)
            IDR_factor (float, optional): 层间位移角的缩放系数
            RIDR_factor (float, optional): 残余层间位移角的缩放系数
            PFA_factor (float, optional): 楼层绝对加速度的缩放系数
            PFV_factor (float, optional): 楼层绝对速度的缩放系数
        
        Note:
        -----
        * 层间位移角、残余层间位移角、楼层加速度、楼层速度的单位应分别为(rad)、(rad)、(g)、(in/s)，
          如果导入的数据不是采用这些单位，则需要调整`IDR_factor`、`RIDR_factor`、`PFA_factor`和
          `PFV_factor`
        * `file_PFV`可以不提供，除非建筑中包含速度敏感型的构件
        """
        self.IDR_data = _read_IDA_file(file_IDR, self.Nstory, gm_num, IDR_factor)
        self.RIDR_data = _read_IDA_file(file_RIDR, self.Nstory, gm_num, RIDR_factor, is_RIDR=True)
        self.PFA_data = _read_IDA_file(file_PFA, self.Nstory, gm_num, PFA_factor)
        self.PFV_data = None
        if file_PFV is not None:
            self.PFV_data = _read_IDA_file(file_IDR, self.Nstory, gm_num, PFV_factor)
        self.IDR_range = IDR_range
        self.RIDR_range = RIDR_range
        self.PFA_range = PFA_range
        self.PFV_range = PFV_range
        LOGGER.success(f'IDA data is imported successfully.')

    def set_seismic_response(self,
            IDR_PSDM: list[VECTOR],
            PFA_PSDM: list[VECTOR],
            RIDR_PSDM: VECTOR = None,
            PFV_PSDM: list[VECTOR] = None,
            IDR_factor: float = 1.0,
            PFA_factor: float = 1.0,
            RIDR_factor: float = 1.0,
            PFV_factor: float = 1.0,
        ):
        """导入概率地震需求模型和倒塌易损性

        Args:
            IDR_PSDM (list[VECTOR]): 各层位移角的PSDM (rad)
            PFA_PSDM (list[VECTOR]): 各层绝对加速度的PSDM (g)
            RIDR_PSDM (VECTOR): 最大残余位移角的PSDM (rad)
            PFV_PSDM (list[VECTOR], optional): 各层绝对速度的PSDM (in/s)
            IDR_factor (float, optional): 位移角需求的缩放系数
            RIDR_factor (float, optional): 残余位移角的缩放系数
            PFA_factor (float, optional): 楼层加速度的缩放系数
            PFV_factor (float, optional): 楼层速度的缩放系数
        
        Note:
        -----
        * FEMA P58采用实际计算得到的需求矩阵来预测地震需求，但是实际IDA计算中，
          每条地震动的强度和计算次数都会动态调整，IDA结果无法与FEMA P58的方法
          适配，因此此处采用概率地震需求模型(PSDM)来预测给定地震强度下的结构地
          震需求，但是仍根据倒塌易损性曲线来计算倒塌概率，因为倒塌级别的地震动
          强度下PSDM会失真。
        * 每种类型EDP的PSDM需传入分别导入`A`、`B`、`sgm`三个参数，并认为
          `ln(DM) = A + B * ln(IM)`，标准差为`sgm`
        * `clps_frag`参数分别为倒塌强度中值`θ`和对数标准差`β`，倒塌概率为：
          `Pc = norm(ln(Sa / exp(θ)) / β, 0, 1)`
        * 位移角、残余位移角、楼层加速度、楼层速度的单位应分别为(rad)、(rad)、
          (g)、(in/s)，如果导入的数据不是采用这些单位，则需要调整`IDR_factor`、
          `PFA_factor`和`PFV_factor`
        """
        self.IDR_PSDM = IDR_PSDM
        self.RIDR_PSDM = RIDR_PSDM
        self.PFA_PSDM = PFA_PSDM
        self.PFV_PSDM = PFV_PSDM
        self.IDR_factor = IDR_factor
        self.RIDR_factor = RIDR_factor
        self.PFA_factor = PFA_factor
        self.PFV_factor = PFV_factor
        LOGGER.success(f'Seismic_response has been defined.')
    
    def set_collapse_prob(self,
            median_clps: float,
            logstd_clps: float,
            collapse_modes: dict[tuple[int], float]
        ):
        """定义倒塌概率（与地震动强度相关）

        Args:
            median_clps (float): 50%倒塌概率对应的中值倒塌强度
            logstd_clps (float): 倒塌强度的对数标准差
            collapse_modes (dict[tuple[int], float]): 倒塌模式及对应的概率
        
        Note:
        -----
        * `collapse_modes`的key为倒塌模式中倒塌的楼层编号(从1开始)，value为概率，
          概率之和应为1
        
        Example:
        -------
        >>> collapse_modes = {
            (1,): 0.6,
            (1, 2): 0.4}
        
        表示有60%的概率发生底层倒塌，有40%的概率发生1-2层倒塌
        """
        self.median_clps = median_clps
        self.logstd_clps = logstd_clps
        if not isclose(sum(collapse_modes.values()), 1):
            raise ValueError('The sum of probability of collapse modes should be 1')
        self.collapse_modes = collapse_modes
        self.account_for_clps = True  # 考虑倒塌
        LOGGER.success(f'Probability of collapse has been defined.')

    def set_demolishment_prob(self,
            median_RIDR: float,
            logstd_RIDR: float
        ):
        """定义拆除概率（与RIDR相关）

        Args:
            median_RIDR (float): 50%拆除概率对应的中值RIDR
            logstd_RIDR (float): RIDR的对数标准差
        """
        self.median_RIDR = median_RIDR
        self.logstd_RIDR = logstd_RIDR
        self.account_for_dm = True  # 考虑拆除
        LOGGER.success(f'Probability of demolishment has been defined.')

    def _simu_clps(self,
            Sa: float,
            is_random: bool = True
        ) -> bool:
        # 模拟倒塌
        if self.account_for_clps is None:
            return False  # 没有定义倒塌易损性，不考虑倒塌
        p = _cdf(np.log(Sa / self.median_clps) / self.logstd_RIDR)
        if is_random:
            clps = np.random.uniform() < p
        else:
            if p > 0.5:
                clps = True
            else:
                clps = False
        return bool(clps)

    def _simu_demolishment(self,
            RIDR: float,
            is_random: bool = True
        ) -> bool:
        # 模拟拆除
        if self.account_for_dm is None:
            return False  # 没有定义拆除概率，不考虑拆除
        p = _cdf(np.log(RIDR / self.median_RIDR) / self.logstd_RIDR)
        if is_random:
            dm = np.random.uniform() < p
        else:
            if p > 0.5:
                dm = True
            else:
                dm = False
        return bool(dm)

    def _simu_IDR(self,
            Sa: float,
            is_random: bool = True
        ) -> np.ndarray:
        """模拟层间位移角需求，ln(IDR) = A + B * ln(Sa)"""
        IDR = np.zeros(self.Nstory)
        for i in range(self.Nstory):
            A, B, logstd = self.IDR_PSDM[i]
            ln_median = A + B * np.log(Sa)
            if is_random:
                IDR[i] = np.exp(_normal(ln_median, logstd))
            else:
                IDR[i] = np.exp(ln_median)
        IDR = np.where(IDR < 0, 0, IDR)
        return IDR * self.IDR_factor
    
    def _simu_RIDR(self,
            Sa: float,
            is_random: bool = True
        ) -> float:
        # ln(RIDR) = A + B * ln(Sa)
        if self.RIDR_PSDM is None:
            LOGGER.warning('RIDR_PSDM is not defined, returning 0% RIDR')
            return 0
        A, B, logstd = self.RIDR_PSDM
        ln_median = A + B * np.log(Sa)
        if is_random:
            RIDR = np.exp(_normal(ln_median, logstd))
        else:
            RIDR = np.exp(ln_median)
        RIDR = np.where(RIDR < 0, 0, RIDR)
        return RIDR * self.RIDR_factor

    def _simu_PFA(self,
            Sa: float,
            is_random: bool = True
        ) -> np.ndarray:
        # ln(PFA) = A + B * ln(Sa)
        PFA = np.zeros(self.Nstory)
        for i in range(self.Nstory):
            A, B, logstd = self.PFA_PSDM[i]
            ln_median = A + B * np.log(Sa)
            if is_random:
                PFA[i] = np.exp(_normal(ln_median, logstd))
            else:
                PFA[i] = np.exp(ln_median)
        PFA = np.where(PFA < 0, 0, PFA)
        return PFA * self.PFA_factor

    def _simu_PFV(self,
            Sa: float,
            is_random: bool = True
        ) -> np.ndarray:
        # ln(PFV) = A + B * ln(Sa)
        PFV = np.zeros(self.Nstory)
        for i in range(self.Nstory):
            A, B, logstd = self.PFV_PSDM[i]
            ln_median = A + B * np.log(Sa)
            if is_random:
                PFV[i] = np.exp(_normal(ln_median, logstd))
            else:
                PFV[i] = np.exp(ln_median)
        PFV = np.where(PFV < 0, 0, PFV)
        return PFV * self.PFV_factor


def _read_IDA_file(
        file_path: str | Path,
        Nstory: int,
        gm_num: int,
        factor: float,
        is_RIDR: bool = False
    ) -> list[np.ndarray]:
    wb = px.load_workbook(file_path)
    data: dict[int, list[np.ndarray]] = {}
    for idx_story in range(Nstory):
        ws = wb.worksheets[idx_story]
        data_story: list[np.ndarray] = []
        for idx_gm in range(gm_num):
            edp = _read_column(ws, 2, idx_gm * 2 + 1)
            im = _read_column(ws, 2, idx_gm * 2 + 2) * factor
            data_story.append(np.column_stack((edp, im)))
        data[idx_story] = data_story
        if is_RIDR:
             break  # RIDR的IDA数据仅需提供各楼层的最大RIDR
    wb.close()
    return data


def _read_column(ws: Worksheet, row: int, col: int) -> np.ndarray:
    """读取worksheet从row行开始，第col列的值，
    row和col从1开始计数
    """
    data = []
    for i in range(row, ws.max_row + 1):
        val = ws.cell(row=i, column=col).value
        if val is not None:
            data.append(float(val))
        else:
            break
    return np.array(data)

