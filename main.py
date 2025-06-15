from src.compenent import Component
from src.building import Building
from config.config import AVAILABLE_COMP


if __name__ == "__main__":

    building = Building(
        name='test_building',
        Nstory=4,
        size=(30, 30),
        heights=[4.3, 4, 4, 4],
        unit='m',
        replacement_cost=1000000)
    building.add_IDAdata(44,
        'data/IDAdata/template_IDR.xlsx',
        'data/IDAdata/template_RIDR.xlsx',
        'data/IDAdata/template_PFA.xlsx',
    )
