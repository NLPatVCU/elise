import os
def clear_dir(path):
    file_list = os.listdir(path)
    for file_name in file_list:
        os.remove(path+file_name)

