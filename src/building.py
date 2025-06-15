from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as px
from openpyxl.worksheet.worksheet import Worksheet
from .compenent import Component
from .unit_convertor import to_foot
from config.config import UNITS_TYPING, LOGGER

class Building:
    def __init__(self,
            name: str,
            Nstory: int,
            size: tuple[float, float],
            heights: list[float],
            unit: UNITS_TYPING,
            replacement_cost: float):
        """定义一栋建筑，并设置基本参数

        Args:
            name (str): 名称
            Nstory (int): 楼层数
            size (tuple[float, float]): 平面尺寸
            heights (list[float]): 层高列表
            unit (UNITS_TYPING): 尺寸和层高的单位
            replacement_cost (float): 重建成本
        
        Notes:
        ------
        所有长度量纲的单位都将转换为英尺(foot)
        """
        self.name = name
        self.Nstory = Nstory
        self.size = to_foot(size, unit)
        self.heights = to_foot(heights, unit)
        self.replacement_cost = replacement_cost
        self.components: list[tuple[Component, float, int, int]] = []
        LOGGER.success(f'Building "{self.name}" is created successfully.')

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
            PFV_factor: float = 1.0
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
        if file_PFV is not None:
            self.PFV_data = _read_IDA_file(file_IDR, self.Nstory, gm_num, PFV_factor)
        LOGGER.success(f'IDA data is imported successfully.')


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


def _plot_IDA_data(IDA_data: list[np.ndarray], edp_name: str):
    label = 'Individual'
    for i, data in enumerate(IDA_data):
        plt.plot(data[:, 0], data[:, 1], color='grey', label=label)
        label = None
    plt.xlabel(edp_name)
    plt.ylabel("IM (g)")
    plt.legend()
    plt.tight_layout()
    plt.show()