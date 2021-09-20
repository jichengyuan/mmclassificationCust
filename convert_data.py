import os
import glob
import re

# 生成train.txt和val.txt

#需要改为您自己的路径
root_dir = "G:\Dataset\imagenette2_tiny"
#在该路径下有train,val，meta三个文件夹
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
meta_dir = os.path.join(root_dir, "meta")

def generate_txt(images_dir,map_dict):
    # 读取所有文件名
    imgs_dirs = glob.glob(images_dir+"\\*\\*")
    # 打开写入文件
    typename = images_dir.split("\\")[-1]
    target_txt_path = os.path.join(meta_dir,typename+".txt")
    print(target_txt_path)
    f = open(target_txt_path,"w")
    # 遍历所有图片名
    for img_dir in imgs_dirs:
        # 获取第一级目录名称
        filename = img_dir.split("\\")[-2]
        f_path = img_dir.split("\\")[-1]
        relate_name = filename+"\\"+f_path
        num = map_dict[filename]
        # 写入文件
        f.write(relate_name+" "+num+"\n")

def get_map_dict():
    # 读取所有类别映射关系
    class_map_dict = {}
    with open(os.path.join(meta_dir,"classmap.txt"),"r") as F:
        lines = F.readlines()
        for line in lines:
            line = line.split("\n")[0]
            filename,cls,num = line.split(" ")
            class_map_dict[filename] = num
    return class_map_dict

if __name__ == '__main__':

    class_map_dict = get_map_dict()

    generate_txt(images_dir=train_dir,map_dict=class_map_dict)

    generate_txt(images_dir=val_dir,map_dict=class_map_dict)