"""
将FEMA P58的易损性数据库中的XML文件转换为JSON格式
"""
from pathlib import Path
import xmltodict
import json


def convert_strings_to_bools(data):
    if isinstance(data, dict):
        return {key: convert_strings_to_bools(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_strings_to_bools(item) for item in data]
    elif isinstance(data, str):
        if data.lower() == "true":
            return True
        elif data.lower() == "false":
            return False
        elif data.lower() in ["none", "None"]:
            return None
        else:
            return data
    else:
        return data


if __name__ == "__main__":
    for file in Path('data/ATCCurves').glob('**/*.xml'):
        print(f'Converting {file} to JSON...')
        with open(file, encoding='utf-8') as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            
        data_dict = convert_strings_to_bools(data_dict)

        with open(f'data/ATCCurves_json/{file.stem}.json', 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
            