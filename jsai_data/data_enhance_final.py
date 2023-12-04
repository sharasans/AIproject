# 导入os模块，用于操作文件系统
import os
import shutil
import sys

import imgaug as ia
import imgaug.augmenters as iaa
import tqdm
import cv2
from tqdm.contrib import tzip
import time

t = time.localtime()
# 定义源文件夹的路径，你可以根据你的实际情况修改
source_dir = "D:/jsai_data/jsai_data"

target_dir = "D:/jsai_data/data_enhance_" + str(t.tm_year) + "_" + str(t.tm_mon) + "_" + str(
    t.tm_mday) + "_" + str(t.tm_hour) + "_" + str(t.tm_min)
# 定义目标文件夹的路径，你可以根据你的实际情况修改
images_ori_dir = target_dir + "/images_ori"
labels_ori_dir = target_dir + "/labels_ori"
images_enhance_dir = target_dir + "/images_enhance"
labels_enhance_dir = target_dir + "/labels_enhance"
train_ori_dir = target_dir + "/train_ori"
val_ori_dir = target_dir + "/val_ori"

images_ori_train_dir = images_ori_dir + "/train"
images_ori_val_dir = images_ori_dir + "/val"
images_ori_test_dir = images_ori_dir + "/test"
images_enhance_train_dir = images_enhance_dir + "/train"
images_enhance_val_dir = images_enhance_dir + "/val"
labels_ori_train_dir = labels_ori_dir + "/train"
labels_ori_val_dir = labels_ori_dir + "/val"
labels_ori_test_dir = labels_ori_dir + "/test"
labels_enhance_train_dir = labels_enhance_dir + "/train"
labels_enhance_val_dir = labels_enhance_dir + "/val"


def makedir():
    os.mkdir(target_dir)
    os.mkdir(images_ori_dir)
    os.mkdir(labels_ori_dir)
    os.mkdir(images_enhance_dir)
    os.mkdir(labels_enhance_dir)
    os.mkdir(train_ori_dir)
    os.mkdir(val_ori_dir)

    os.mkdir(images_ori_train_dir)
    os.mkdir(images_ori_val_dir)
    os.mkdir(images_ori_test_dir)
    os.mkdir(images_enhance_train_dir)
    os.mkdir(images_enhance_val_dir)

    os.mkdir(labels_ori_train_dir)
    os.mkdir(labels_ori_val_dir)
    os.mkdir(labels_ori_test_dir)
    os.mkdir(labels_enhance_train_dir)
    os.mkdir(labels_enhance_val_dir)


def separate(source_dir, train_image_dir, train_labels_dir, val_image_dir, val_labels_dir, test_image_dir,
             test_labels_dir):
    # 获取源文件夹中的所有文件名，存入一个列表
    file_names = os.listdir(source_dir)

    # 过滤掉非图片和非txt文件，只保留符合yolo格式的文件
    file_names = [f for f in file_names if f.endswith(".jpg") or f.endswith(".txt")]

    # 导入random模块，用于生成随机数
    import random

    # 定义一个函数，用于将文件名按照图片和标签分成两个列表
    def split_files(files):
        # 创建两个空列表，分别存储图片和标签文件名
        image_files = []
        label_files = []
        # 遍历文件列表
        for f in files:
            # 判断文件是图片还是txt
            if f.endswith(".jpg"):
                # 将图片文件名添加到image_files列表
                image_files.append(f)
            elif f.endswith(".txt"):
                # 将标签文件名添加到label_files列表
                label_files.append(f)
        # 返回两个列表
        return image_files, label_files

    # 调用函数，将file_names分成两个列表
    image_files, label_files = split_files(file_names)

    # 定义一个函数，用于打乱两个列表，并保证图片和标签的对应关系
    def shuffle_files(image_files, label_files):
        # 创建一个空列表，用于存储打乱后的图片和标签文件名
        shuffled_files = []
        # 获取图片和标签文件的数量，应该相同
        num = len(image_files)
        # 生成一个随机数列表，作为打乱的索引
        indices = list(range(num))
        random.shuffle(indices)
        # 遍历随机数列表
        for i in indices:
            # 根据索引获取对应的图片和标签文件名
            image_file = image_files[i]
            label_file = label_files[i]
            # 将图片和标签文件名添加到shuffled_files列表
            shuffled_files.append(image_file)
            shuffled_files.append(label_file)
        # 返回打乱后的文件列表
        return shuffled_files

    # 调用函数，将image_files和label_files打乱
    shuffled_files = shuffle_files(image_files, label_files)

    # 计算每个文件夹应该分配的文件数量，按照8:1:1的比例
    total = len(shuffled_files)
    train_num = int(int(total * 0.7) / 2) * 2
    val_num = int(int(total * 0.2) / 2) * 2
    test_num = total - train_num - val_num

    # 将文件名分配到三个列表中，分别对应训练集，验证集和测试集
    train_files = shuffled_files[:train_num]
    val_files = shuffled_files[train_num:train_num + val_num]
    test_files = shuffled_files[train_num + val_num:]

    # 定义一个函数，用于复制文件到目标文件夹
    import shutil

    def copy_files(files, target_dir):
        # 遍历文件列表
        for f in files:
            # 获取文件的完整路径
            source_path = os.path.join(source_dir, f)
            # 获取目标文件夹的完整路径
            target_path = os.path.join(target_dir, f)
            # 复制文件
            shutil.copy(source_path, target_path)

    # 定义一个函数，用于根据文件类型分配到不同的文件夹
    def distribute_files(files, image_dir, labels_dir):
        # 遍历文件列表
        from tqdm import tqdm
        for f in tqdm(files):
            # 判断文件是图片还是txt
            if f.endswith(".jpg"):
                # 复制图片到image文件夹
                copy_files([f], image_dir)
            elif f.endswith(".txt"):
                # 复制txt到labels文件夹
                copy_files([f], labels_dir)

    # 调用函数，将文件复制到对应的文件夹
    distribute_files(train_files, train_image_dir, train_labels_dir)
    distribute_files(val_files, val_image_dir, val_labels_dir)
    distribute_files(test_files, test_image_dir, test_labels_dir)


def copy_allfiles(src, dest):
    # src:原文件夹；dest:目标文件夹
    src_files = os.listdir(src)
    from tqdm import tqdm
    for file_name in tqdm(src_files):
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)


aug = iaa.SomeOf((0, 3), [
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.02)),  # 裁剪
    iaa.Fliplr(0.5),  # 水平翻转
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # 锐化，增强图像的边缘和细节
    iaa.AllChannelsHistogramEqualization(),
    iaa.GammaContrast(gamma=(0.5, 2.0), per_channel=0.5),
    iaa.LinearContrast(alpha=(0.5, 2.0), per_channel=0.5)
])


def enhance(ori_path, images_enhance_path, labels_enhance_path):
    # 定义图片和标注文件的路径
    ori_path = ori_path + '/'
    images_enhance_path = images_enhance_path + '/'
    labels_enhance_path = labels_enhance_path + '/'
    img_path = ori_path  # 图片文件夹
    txt_path = img_path  # 标注文件夹和图片文件夹相同
    img_list = os.listdir(img_path)  # 图片列表
    txt_list = [img_name[:-4] + ".txt" for img_name in img_list]  # 标注列表
    img_list_h = [img_name[:-4] + ".jpg" for img_name in img_list]  # 标注列表
    img_list = img_list_h
    img_path_en = images_enhance_path  # 图片文件夹
    txt_path_en = labels_enhance_path

    aug_times = 20
    # 遍历每张图片和对应的标注文件
    for img_name, txt_name in tzip(img_list, txt_list):
        # 读取图片和标注
        img = cv2.imread(img_path + img_name)
        with open(txt_path + txt_name, "r") as f:
            labels = f.readlines()
        flag16 = 0
        flag3 = 0
        flagelse = 0
        flag7 = 0
        flag0 = 0
        for label in labels:
            label = label.strip().split()
            cls, x, y, w, h = label
            if cls == '7':
                flag7 = 1
                break
            elif cls == '1' or cls == '6':
                flag16 = 1
            elif cls == '0':
                flag0 = 1
            elif cls == '3':
                flag3 = 1
            else:
                flagelse = 1
        if flag16 == 0 and flagelse == 1:
            aug_times = 107
        elif flag16 == 1 and flagelse == 1:
            aug_times = 3
        elif flag16 == 1 and flagelse == 0:
            aug_times = 0
        if flag0 == 1:
            aug_times = 7
        if flag3 == 1:
            aug_times = 27
        if flag7 == 1:
            aug_times = 63
        bbs = []
        for label in labels:
            label = label.strip().split()
            cls, x, y, w, h = label  # 类别，中心坐标，宽度，高度
            w = float(w)
            h = float(h)
            x = float(x)
            y = float(y)
            x1 = (x - w / 2) * img.shape[1]  # 左上角x坐标
            y1 = (y - h / 2) * img.shape[0]  # 左上角y坐标
            x2 = (x + w / 2) * img.shape[1]  # 右下角x坐标
            y2 = (y + h / 2) * img.shape[0]  # 右下角y坐标
            bbs.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=cls))
        bbs = ia.BoundingBoxesOnImage(bbs, shape=img.shape)
        i = 1
        # 对每张图片进行数据增强
        while i <= aug_times:
            i = i + 1
            # 应用数据增强
            flag = 1
            img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
            for bb in bbs_aug.bounding_boxes:
                if bb.x1 < -0.1 or bb.x2 < -0.1 or bb.y1 < -0.1 or bb.y2 < -0.1 or bb.x1 > img_aug.shape[
                    1] + 0.1 or bb.x2 > img_aug.shape[1] + 0.1 or bb.y1 > img_aug.shape[0] + 0.1 or bb.x1 > \
                        img_aug.shape[1] + 0.1:
                    i = i - 1
                    flag = 0
                    break
            if flag == 0:
                continue
            # 保存增强后的图片
            img_aug_name = img_name[:-4] + "_aug" + str(i) + ".jpg"
            cv2.imwrite(img_path_en + img_aug_name, img_aug)
            # 保存增强后的标注
            txt_aug_name = txt_name[:-4] + "_aug" + str(i) + ".txt"
            with open(txt_path_en + txt_aug_name, "w") as f:
                for bb in bbs_aug.bounding_boxes:
                    # 将标注转换回yolo格式
                    cls = bb.label  # 类别
                    x = (bb.x1 + bb.x2) / 2 / img_aug.shape[1]  # 中心x坐标
                    y = (bb.y1 + bb.y2) / 2 / img_aug.shape[0]  # 中心y坐标
                    w = (bb.x2 - bb.x1 - 1) / img_aug.shape[1]  # 宽度
                    h = (bb.y2 - bb.y1 - 1) / img_aug.shape[0]  # 高度
                    f.write(f"{cls} {x} {y} {w} {h}\n")  # 写入文件


def counting_label(label_path, label_txt):
    label_path = label_path + '/'
    count_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
    txt_list = os.listdir(label_path)
    txt_count = len(txt_list)
    for txt_name in txt_list:
        with open(label_path + txt_name, "r") as f:
            labels = f.readlines()
            for label in labels:
                label = label.strip().split()
                cls, x, y, w, h = label
                count_dict[cls] = count_dict[cls] + 1
    with open(target_dir + '/' + label_txt, "w") as f:
        for k, v in count_dict.items():
            f.write(str(k) + ' ' + str(v) + '\n')
        f.write(str(txt_count))
    print(label_path)
    print(count_dict)
    print(txt_count)
    max_key = max(count_dict, key=count_dict.get)
    min_key = min(count_dict, key=count_dict.get)
    if count_dict[min_key] == 0:
        print("not good enough")
        return 0, txt_count
    if count_dict[max_key] / count_dict[min_key] > 5:
        print("not good enough")
        return 0, txt_count
    return 1, txt_count


def restart_program():
    os.system("python D:/jsai_data/data_enhance_final.py")


if __name__ == '__main__':
    print('makedir')
    makedir()
    print('seperate train val and test')
    separate(source_dir, images_ori_train_dir, labels_ori_train_dir, images_ori_val_dir, labels_ori_val_dir,
             images_ori_test_dir, labels_ori_test_dir)
    print('copy files for further use')
    copy_allfiles(images_ori_train_dir, train_ori_dir)
    copy_allfiles(labels_ori_train_dir, train_ori_dir)
    copy_allfiles(images_ori_val_dir, val_ori_dir)
    copy_allfiles(labels_ori_val_dir, val_ori_dir)
    print('start data enhance')
    enhance(train_ori_dir, images_enhance_train_dir, labels_enhance_train_dir)
    enhance(val_ori_dir, images_enhance_val_dir, labels_enhance_val_dir)
    copy_allfiles(images_ori_train_dir, images_enhance_train_dir)
    copy_allfiles(images_ori_val_dir, images_enhance_val_dir)
    copy_allfiles(labels_ori_train_dir, labels_enhance_train_dir)
    copy_allfiles(labels_ori_val_dir, labels_enhance_val_dir)
    et, ct = counting_label(labels_enhance_train_dir, "train_labels_count.txt")
    ev, cv = counting_label(labels_enhance_val_dir, "val_labels_count.txt")
    if et == 1 and ev == 1 and ct / cv <= 8:
        print('complete')
    else:
        restart_program()
