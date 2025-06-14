from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as px
from openpyxl.worksheet.worksheet import Worksheet
from .compenent import Component


class Building:
    def __init__(self,
                 name: str,
                 Nstory: int,
                 Nbay: int,
                 heights: list[float]):
        self.name = name
        self.Nstory = Nstory
        self.Nbay = Nbay
        self.heights = heights

    def set_replacement_cost(self, replacement_cost: float):
        self.replacement_cost = replacement_cost
    
    def add_component(self, component: Component, num: float):
        ...
    
    def add_IDAdata(self,
                    file_IDR: str | Path,
                    file_PFA: str | Path,
                    file_PFV: str | Path = None,
                    file_RIDR: str | Path = None,
                    plot: bool = False):
        IDR_data = _parse_IDA_file(file_IDR)
        PFA_data = _parse_IDA_file(file_PFA)
        if file_PFV is not None:
            PFV_data = _parse_IDA_file(file_PFV)
        if file_RIDR is not None:
            RIDR_data = _parse_IDA_file(file_RIDR)
        if plot:
            _plot_IDA_data(IDR_data, "IDR (rad)")
            _plot_IDA_data(PFA_data, "PFA (g)")
            if file_PFV is not None:
                _plot_IDA_data(PFV_data, "PFV (mm/s)")
            if file_RIDR is not None:
                _plot_IDA_data(RIDR_data, "RIDR (rad)")


def _parse_IDA_file(file_path: str | Path) -> list[np.ndarray]:
    wb = px.load_workbook(file_path)
    ws = wb.active
    gm_num = 0  # 地震动数量
    row = 2  # 起始行
    IDA_data: list[np.ndarray] = []
    while True:
        edp = _get_column(ws, row, gm_num * 2 + 1)
        im = _get_column(ws, row, gm_num * 2 + 2)
        if not edp:
            break
        IDA_data.append(np.column_stack((edp, im)))
        gm_num += 1
    return IDA_data


def _get_column(ws: Worksheet, row: int, col: int):
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
    return data


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