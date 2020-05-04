"""
16.412 Intent Inference GC | parse_data.py
Converts text representation of CARLA sim data into .json files for use with
Constant Velocity Method (CVM) trajectory prediction.
Author: Abbie Lee (abbielee@mit.edu)
"""

import json

def to_dicts(fname, id = 0, frame = 0):
    f = open(fname, mode='r')

    labels = []
    timestamped_dicts = []

    for line in f:
        if len(labels) == 0:
            dividers = [i for i, char in enumerate(line) if char == "|"]
            dividers = [-2] + dividers
            labels = [line[dividers[i]+2:dividers[i+1]-1] for i in range(len(dividers)-1)]
            continue

        ts_dict = {"timestamp": None, "object_list": [], "frame": frame, "size": 0}

        dividers = [i for i, char in enumerate(line) if char == "|"]
        dividers = [-2] + dividers
        elements = [line[dividers[i]+2:dividers[i+1]-1] for i in range(2)]

        ts_dict["timestamp"] = round(float(elements[0]), 2)

        new_obj = {"position": [], "id": id}
        pos = [round(float(e), 2) for e in elements[1].split(',')]
        new_obj["position"] = pos[:2]

        ts_dict["object_list"].append(new_obj)

        ts_dict["size"] = len(ts_dict["object_list"])

        timestamped_dicts.append(ts_dict)

    return timestamped_dicts

def dict_to_json(ts_dicts, out_path):
    counter = 1
    for d in ts_dicts:
        with open(out_path + str(counter) + ".json", mode="w") as json_file:
            json.dump(d, json_file)
        counter += 1

if __name__=="__main__":
    data_path = "data/raw/"
    out_path = "data/CARLA_dataset/data/"
    file = "myrecording.txt"

    ts_dicts = to_dicts(data_path + file)
    dict_to_json(ts_dicts, out_path)
