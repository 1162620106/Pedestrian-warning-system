import os
from yolo import YOLO
from PIL import Image

test_img_path = 'C:/AIoT/Projects/keras-yolo3-master/VOCtest/VOC2007/JPEGImages'
img_list = os.listdir(test_img_path)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表

yolo_obj = YOLO()

for each_img in img_list:
    img_name, _ = each_img.split('.')
    image = Image.open(test_img_path + '/' + each_img)
    yolo_obj.detect_image(image, img_name)

yolo_obj.close_session()