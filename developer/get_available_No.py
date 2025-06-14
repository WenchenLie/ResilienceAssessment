from pathlib import Path


if __name__ == "__main__":
    No_list = []
    for file in Path('data/ATCCurves_json').iterdir():
        if file.suffix == '.json':
            No = file.stem
            No_list.append(repr(No))

    print(No_list)
    No_list = ',\n'.join(No_list)
    with open('temp/data.txt', 'w') as f:
        f.writelines(No_list)