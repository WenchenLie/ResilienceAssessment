from src.compenent import Component
from src.building import Building


if __name__ == "__main__":
    comp = Component("B1031.001")
    data = []
    print(comp._get_cost(20, 2, is_random=True))
    
    # building = Building('test_building', 4, 3, [4300, 4000, 4000, 4000])
    # building.add_IDAdata(
    #     'data/IDAdata/IDA曲线_IDR.xlsx',
    #     'data/IDAdata/IDA曲线_PFA.xlsx',
    #     plot=True
    # )
