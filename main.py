from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.compenent import Component
from src.building import Building
from src.consequance_estimate import consequance_estimate
from src.post_processing import post_processing


def calculate(
    output_dir: str | Path,
    hazard_curve: np.ndarray,
):
    """计算建筑震后损失

    Args:
        output_dir (str | Path): 输出文件夹路径
        hazard_curve (np.ndarray): 地震危险性曲线
    """
    Nstory = 4
    x_size, y_size = 36.6, 24.4
    area = x_size * y_size
    n_MC = 1000
    Sa_ls = np.linspace(0.01, 4, 100)
    building = Building(
        name='test_building',
        Nstory=Nstory,
        size=(x_size, y_size),
        heights=[4.3, 4, 4, 4],
        unit='m',
        replacement_cost=12_000_000,
        replacement_time=720,
        occupancy='Commercial Office',
        n_MC=n_MC,
        Sa_ls=Sa_ls
    )

    shear_connection = Component('B1031.001', 'S')
    column_base = Component('B1031.011b', 'S')
    column_splices = Component('B1031.021b', 'S')
    # moment_connection_one_side = Component('B1035.021')
    moment_connection_both_side = Component('B1035.031', 'S')
    crtain_wall = Component('B2022.001', 'NS')
    stair = Component('C2011.011b', 'NS')
    suspended_ceiling = Component('C3032.003a', 'C')
    independent_pendant_lighting = Component('C3034.001', 'C')
    cold_or_hot_potable = Component('D2021.011a', 'C')
    sanitary_waste_piping = Component('D2031.011b', 'C')
    HVAC = Component('D3041.001a', 'C')
    modular_office_work_stations = Component('E2022.001', 'C')
    unsecured_fragile_objects_on_shelves = Component('E2022.010', 'C')
    electronic_equipment_on_wall_mount_brackets = Component('E2022.021', 'C')
    desktop_electronics = Component('E2022.022', 'C')
    bookcase_2shelves = Component('E2022.102b', 'C')
    # Ref: Seismic fragility and loss estimation of self-centering steel braced frames under mainshock-aftershock sequences

    for story in range(1, Nstory + 1):
        top_floor = story + 1
        bot_floor = story
        building.add_component(shear_connection, 24, story=story)
        building.add_component(moment_connection_both_side, 4, story=story)
        building.add_component(crtain_wall, 58, story=story)
        building.add_component(stair, 2, story=story)
        building.add_component(suspended_ceiling, area * 10.7639104 / 250, floor=top_floor)
        building.add_component(independent_pendant_lighting, 50, floor=top_floor)
        building.add_component(cold_or_hot_potable, 2, floor=top_floor)
        building.add_component(sanitary_waste_piping, 1, floor=top_floor)
        building.add_component(HVAC, 5, floor=top_floor)
        building.add_component(modular_office_work_stations, 50, floor=bot_floor)
        building.add_component(unsecured_fragile_objects_on_shelves, 50, floor=bot_floor)
        building.add_component(electronic_equipment_on_wall_mount_brackets, 20, floor=bot_floor)
        building.add_component(desktop_electronics, 40, floor=bot_floor)
        building.add_component(bookcase_2shelves, 50, floor=bot_floor)
    building.add_component(column_base, 35, 1)
    building.add_component(column_splices, 6, 3)

    building.set_expanded_EDPmat(r"H:\results_TSSCB_Frame\IDA2\MRF4S8_base_frag\EDP_matrix.csv")
    building.set_demolishment_prob(median_RIDR=0.005, logstd_RIDR=0.3)
    building.set_collapse_prob(median_clps=3.0, logstd_clps=0.4,
                               collapse_modes={
                                   (1, 2, 3, 4): 0.6,
                                   (1,): 0.4})

    consequance_estimate(
        building,
        hazard_curve,
        root=output_dir,
        parallel=12
    )


if __name__ == "__main__":
    from viztracer import VizTracer
    root = Path('Results')
    hazard_curve = np.loadtxt(r'data\hazard_curves\0.83.txt')
    # with VizTracer(log_gc=True, log_async=True, output_file='results.json', max_stack_depth=1000000):
    calculate(root, hazard_curve)
    post_processing(root)
