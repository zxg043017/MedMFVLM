import os
import shutil

# 定义文件夹路径
path = "/mnt/dgx-server/wqruan/M3D/Data/data/M3D_Cap/ct_quizze"
folders = os.listdir(path)


# 移动B和C文件夹中的文件夹到A目录下
for folder in folders:
    folder_path = os.path.join(path, folder)
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            shutil.move(sub_folder_path, path)

# 删除B和C文件夹
for folder in folders:
    folder_path = os.path.join(path, folder)
    if "ct_quizze_" in folder_path:
        shutil.rmtree(folder_path)

print("任务完成")
