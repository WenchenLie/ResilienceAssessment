from src.compenent import Component
from src.building import Building
from config.config import AVAILABLE_COMP


if __name__ == "__main__":

    x_size, y_size = 36.6, 24.4
    area = x_size * y_size

    building = Building(
        name='test_building',
        Nstory=4,
        size=(x_size, y_size),
        heights=[4.3, 4, 4, 4],
        unit='m',
        replacement_cost=1000000)
    
    shear_connection = Component('B1031.001')
    column_base = Component('B1031.011b')
    column_splices = Component('B1031.021b')
    # moment_connection_one_side = Component('B1035.021')
    moment_connection_both_side = Component('B1035.031')
    crtain_wall = Component('B2022.001')
    stair = Component('C2011.011b')
    suspended_ceiling = Component('C3032.003a')
    independent_pendant_lighting = Component('C3034.001')
    cold_or_hot_potable = Component('D2021.011a')
    sanitary_waste_piping = Component('D2031.011b')
    HVAC = Component('D3041.001a')
    modular_office_work_stations = Component('E2022.001')
    unsecured_fragile_objects_on_shelves = Component('E2022.010')
    electronic_equipment_on_wall_mount_brackets = Component('E2022.021')
    desktop_electronics = Component('E2022.022')
    bookcase_2shelves = Component('E2022.102b')
    # Ref: Seismic fragility and loss estimation of self-centering steel braced frames under mainshock-aftershock sequences

    for story in range(1, 5):
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

    building.set_seismic_response(
        IDR_PSDM=[
            [-3.024, 1.046, 0.4],
            [-3.024, 1.046, 0.4],
            [-3.024, 1.046, 0.4],
            [-3.024, 1.046, 0.4]
        ],
        RIDR_PSDM=[
            [-4.291, 2.178, 0.4],
            [-4.291, 2.178, 0.4],
            [-4.291, 2.178, 0.4],
            [-4.291, 2.178, 0.4]
        ],
        PFA_PSDM=[
            [0.384, 0.731, 0.4],
            [0.384, 0.731, 0.4],
            [0.384, 0.731, 0.4],
            [0.384, 0.731, 0.4]
        ],
        clps_frag=[2.0, 0.4]
    )
    building.set_demolishment_prob(
        median_RIDR=0.01,
        logstd=0.3
    )

    

