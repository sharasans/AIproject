# 导入os模块，用于操作文件系统
import os
import tqdm
import imgaug as ia
import imgaug.augmenters as iaa

aug = iaa.Sequential([
    iaa.Affine(rotate=(-3, 3)),  # 旋转
    iaa.Crop(percent=(0, 0.02)),  # 裁剪
    iaa.Fliplr(0.5),  # 水平翻转
    iaa.GaussianBlur(sigma=(0, 0.5)),  # 高斯模糊
    iaa.Multiply((0.8, 1.2)),  # 亮度变化
    iaa.AddToHueAndSaturation((-20, 20)),  # 色调和饱和度变化
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # 添加高斯噪声，增加图像的噪声水平
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # 锐化，增强图像的边缘和细节
    iaa.PerspectiveTransform(scale=(0.01, 0.05))  # 透视变换，改变图像的视角
])
# 定义源文件夹的路径，你可以根据你的实际情况修改
source_dir = "E:/人工智能综合课设/jsai_data/jsai_data"

# 定义目标文件夹的路径，你可以根据你的实际情况修改
image_dir = "E:/人工智能综合课设/jsai_data/images"
labels_dir = "E:/人工智能综合课设/jsai_data/labels"

# 在目标文件夹下创建三个子文件夹，分别对应训练集，验证集和测试集
train_image_dir = os.path.join(image_dir, "train")
val_image_dir = os.path.join(image_dir, "val")
test_image_dir = os.path.join(image_dir, "test")

train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")
test_labels_dir = os.path.join(labels_dir, "test")

# 创建目标文件夹，如果已经存在则跳过
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)

os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

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
train_num = int(int(total * 0.8) / 2) * 2
val_num = int(int(total * 0.1) / 2) * 2
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

# 打印完成的信息
print("Done! The files have been copied to the following directories:")
print("Train images: " + train_image_dir)
print("Train labels: " + train_labels_dir)
print("Val images: " + val_image_dir)
print("Val labels: " + val_labels_dir)
print("Test images: " + test_image_dir)
print("Test labels: " + test_labels_dir)
