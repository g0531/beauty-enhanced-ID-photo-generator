from PIL import Image
from removebg import RemoveBg
from pathlib import Path
import base64
import cv2
import numpy as np
from resize import bicubic_interpolation

# 修改照片背景色
def change_bgd(image, bg_color):
    # 在程序当前目录创建一个error.log文件来保存错误信息(必要参数)
    Path('error.log').touch()
    api_key = "Ku18iyqyvEUg7uUagXKXw1hH"
    img_no_bgd_path = r'E:\数字图像处理\project\BeautyCamera-master\image_no_bgd\image'
    rmbg = RemoveBg(api_key, 'error.log')
    # 执行后会在 read_path 同级文件夹内生成一张 xxx_no_bg.png 的图片
    # 将 cv2 图像转换为字节流
    _, stream = cv2.imencode('.jpg', image)
    # 将字节流转换为 base64 编码字符串
    base64_image = base64.b64encode(stream)
    # 将 base64 编码字符串转换为 Unicode 字符串
    base64_image_string = base64_image.decode('utf-8')
    rmbg.remove_background_from_base64_img(base64_image_string,new_file_name=img_no_bgd_path + "_no_bg.png")
    img_no_bg = Image.open(img_no_bgd_path + '_no_bg.png')
    # 创建一个新的图像，RGB代表真色彩，3通道，
    # color可以为颜色英文名 red 或是 十六进制颜色代码 #00FF00
    new_img = Image.new('RGB', img_no_bg.size, color=bg_color)
    # 将没有背景的图像粘贴到含背景色的图像上
    new_img.paste(img_no_bg, (0, 0, *img_no_bg.size), img_no_bg)
    new_img = cv2.cvtColor(np.asarray(new_img),cv2.COLOR_RGB2BGR)
    return new_img


# 修改照片尺寸：
# （1）计算标准图片的宽高比例ratio1，计算原图的宽高比例ratio2，ratio = w/h
# （2）ratio2 >= ratio1，进行左右裁剪，使w变小达到标准ratio1的比例
# （3）ratio2 < ratio1，进行下裁剪，使h变小达到标准ratio1的比例（不进行上裁剪就是怕人头被剪到）
# （4）resize到标准的尺寸
def change_size(image, ori_width, ori_height, width, height):
    original_ratio = ori_width/ori_height
    new_ratio = width/height
    if original_ratio >= new_ratio:
        image = crop_l_r(image, new_ratio, ori_width, ori_height)
    else:
        image = crop_d(image, new_ratio, ori_width, ori_height)
    new_img = cv2.resize(image,(width, height),interpolation=cv2.INTER_AREA)
    # new_img = bicubic_interpolation(image,(height,width))
    return new_img 

def crop_l_r(image,new_ratio, ori_width, ori_height):
    """
    左右裁剪
    原比例 > 新比例
    """
    crop_w = int(new_ratio * ori_height)
    hide_w = (ori_width - crop_w) // 2
    new_img = image[:, hide_w:ori_width - hide_w]
    return new_img

def crop_d(image, new_ratio, ori_width, ori_height):
    """
    图片下裁剪
    原比例 < 新比例
    """
    crop_h = int(ori_width / new_ratio)
    hide_h = ori_height - crop_h
    new_img = image[:ori_height-hide_h, :]
    return new_img

# path = r'E:\数字图像处理\project\BeautyCamera-master\3.jpg'
# img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
# new_image = change_bgd(img, bg_color='red')
# # new_image.show()

# cv2.imshow("new_image",new_image)
# cv2.waitKey(0)
