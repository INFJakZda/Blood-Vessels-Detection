import os

if __name__ == "__main__":
    data_catalog = "data"
    files_names = [x for x in os.listdir(data_catalog) if os.path.isfile(os.path.join(data_catalog,x))]
    files_names.sort()
    out_catalog = "out"
    for file_name in files_names:
        print(file_name)
