# import os
# import zipfile
# from concurrent.futures import ThreadPoolExecutor

# def unzip_files_to_same_folder(folder_path, output_path):
#     # 获取文件夹中的所有文件名
#     with ThreadPoolExecutor() as executor:
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
            
#             # 检查文件是否是zip文件
#             if zipfile.is_zipfile(file_path):
#                 # 创建一个与zip文件同名的文件夹
#                 extract_to = os.path.join(output_path, os.path.splitext(filename)[0])
#                 if not os.path.exists(extract_to):
#                     os.makedirs(extract_to)
                
#                 # 解压到指定文件夹
#                 with zipfile.ZipFile(file_path, 'r') as zip_ref:
#                     zip_ref.extractall(extract_to)
#                     print(f"{filename} 解压到 {extract_to}")

# # 输入文件夹路径
# folder_path = 'Data/data/datasets--GoodBaiBai88--M3D-Cap/snapshots/d8c62b687c3f1028fa66c8e8c524fe434544a0b2/M3D_Cap/ct_case'
# output_path = '/home/wqruan/M3D/Data/data/M3D_Cap/ct_case/'
# # 解压文件夹中的所有zip文件
# unzip_files_to_same_folder(folder_path, output_path)


import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

def unzip_file(file_path, extract_to):
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"{os.path.basename(file_path)} 解压到 {extract_to}")

def unzip_files_in_folder(folder_path, output_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.zip')]
    with ThreadPoolExecutor() as executor:
        for file_path in files:
            extract_to = os.path.join(output_path, os.path.splitext(os.path.basename(file_path))[0])
            if not os.path.exists(extract_to):
                os.makedirs(extract_to)
            executor.submit(unzip_file, file_path, extract_to)

# 输入文件夹路径
folder_path = '/raid/export/wqruan/M3D/Data/data/datasets--GoodBaiBai88--M3D-Cap/snapshots/d8c62b687c3f1028fa66c8e8c524fe434544a0b2/M3D_Cap/ct_quizze'
output_path = '/raid/export/wqruan/M3D/Data/data/M3D_Cap/ct_quizze/'
# 解压文件夹中的所有zip文件
unzip_files_in_folder(folder_path, output_path)
