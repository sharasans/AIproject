import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import tqdm
from tqdm.contrib import tzip

count_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
# 定义图片和标注文件的路径
img_path = "E:/人工智能综合课设/jsai_data/labels/val/"  # 图片文件夹
txt_path = img_path  # 标注文件夹和图片文件夹相同
img_list = os.listdir(img_path)  # 图片列表
txt_list = [img_name[:-4] + ".txt" for img_name in img_list]  # 标注列表
img_list_h = [img_name[:-4] + ".jpg" for img_name in img_list]  # 标注列表
img_list = img_list_h

for img_name, txt_name in tzip(img_list, txt_list):
    with open(txt_path + txt_name, "r") as f:
        labels = f.readlines()
        for label in labels:
            label = label.strip().split()
            cls, x, y, w, h = label
            count_dict[cls] = count_dict[cls] + 1
print(count_dict)
