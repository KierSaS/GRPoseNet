from pope_model_api import *
import cv2
import numpy as np 
from loguru import logger


#这段代码是一个Python脚本，主要用于加载模型、生成掩码并渲染图像。其中的函数render_mask用于将一组掩码渲染成彩色掩码图像。
# 然后，使用SAM模型生成掩码，并将原始图像与渲染后的掩码图像叠加，得到最终的渲染图像。最后将结果保存到指定的文件路径

#这里的操作是生成所有物体的掩码及渲染图像

def render_mask(masks):
    if len(masks) == 0:
        return None

    #  创建一个全零数组，形状与第一个掩码的形状相同
    res = np.zeros([masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3])

    # 对掩码列表进行排序，按掩码面积大小降序排列
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    # 遍历排序后的掩码列表
    for mask in sorted_masks:
        m = mask["segmentation"]
        res[:, :, 0][m] = np.random.randint(0, 255)
        res[:, :, 1][m] = np.random.randint(0, 255)
        res[:, :, 2][m] = np.random.randint(0, 255)
    res = res.astype(np.uint8)#  # 将结果数组转换为无符号整型
    return res

if __name__ == "__main__":

    # 获取模型检查点和模型类型信息
    ckpt, model_type = get_model_info("h")#对应三个sam的模型尺寸

    # 根据模型类型选择相应的模型，初始化模型对象
    sam = sam_model_registry[model_type](checkpoint=ckpt)


    DEVICE = "cuda"
    sam.to(device=DEVICE)
    #  # 使用SAM模型创建自动掩码生成器对象
    MASK_GEN = SamAutomaticMaskGenerator(sam)

    # 记录日志，显示加载的SAM模型路径
    logger.info(f"load SAM model from {ckpt}")
    
    full_file_name = "/home/wubin/code/POPE/data/demos/5.png"
    image = cv2.imread(full_file_name)

    masks = MASK_GEN.generate(image)
    color_mask = render_mask(masks)


    thres = 0.75
    #    # 将原始图像与彩色掩码图像叠加，得到最终渲染图像
    render_img = (image * thres + color_mask * (1 - thres)).astype(np.uint8)

    # 生成保存结果的目标文件名
    DEST_image = full_file_name.replace(".png", "5_mask.png")

    # # 如果渲染图像不为空，则保存结果图像
    if render_img is not None:
        cv2.imwrite(DEST_image, render_img)
        logger.info(f"result is saved at: {DEST_image}")