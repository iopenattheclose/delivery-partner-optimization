import dill
import os
import pandas as pd

def read_data_and_store_as_dill(file_path, output_file_path):
    try:
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            data = pd.read_excel(file_path)

        with open(output_file_path, "wb") as f:
            dill.dump(data, f)

        print("Data stored successfully as dill file.")
    except Exception as e:
        print("Error reading or storing data:", e)



if __name__ == '__main__':
    input_file = "data/store_info.csv"
    output_file = "data/store_info.dill"
    read_data_and_store_as_dill(input_file, output_file)
