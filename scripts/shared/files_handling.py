import os
import json

def append_change_to_file(output_file, data,data_name):
    data_file = f"{output_file}.raw_data.json"
    # load existing data from the file
    if os.path.exists(data_file):
         with open(data_file, "r") as fin:
            existing_data = json.load(fin)
    else:
        raise FileNotFoundError(f"Data file {data_file} does not exist.")
    # append augmented data
    if data_name in existing_data:
        print("Augmented data already exists. It will be overwritten.")
    existing_data[data_name] = data
    # save updated data back to the file
    with open(data_file, "w") as fout:
        json.dump(existing_data, fout, indent=4)
        print(f"Augmented data appended and saved into {data_file}.")

def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data