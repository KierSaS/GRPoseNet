import torch
import numpy
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from pathlib import Path  # 导入Path类，用于处理文件路径
from PIL import Image
import cv2 #opencv库
from torchvision import transforms

#定义了一个plot.pca的可视化PCA降维结果的方法
def plot_pca( pca_image: numpy.ndarray, save_dir: str, last_components_rgb: bool = False,
             save_resized=False, save_prefix: str = ''):
    """
    finding pca of a set of images.
    :param pil_image: The original PIL image.
    :param pca_image: A numpy tensor containing pca components of the image. HxWxn_components
    :param save_dir: if None than show results.
    :param last_components_rgb: If true save last 3 components as RGB image in addition to each component separately.
    :param save_resized: If true save PCA components resized to original resolution.
    :param save_prefix: optional. prefix to saving
    :return: a list of lists containing an image and its principal components.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    # 将保存目录路径转换为Path对象，并确保目录存在

    h,w = pca_image.shape

    # 将PCA图像归一化处理，使其值范围在0到255之间
    comp = pca_image
    comp_min = comp.min(axis=(0, 1)) # 沿着指定轴计算最小值
    comp_max = comp.max(axis=(0, 1))# 沿着指定轴计算最大值

    comp_img = (comp - comp_min) / (comp_max - comp_min)#归一化公式，x-xmin/xmax-xmin,处理后数值在0-1之间

    comp_img = (comp_img * 255).astype(np.uint8) # 转换为8位无符号整数类型

    # 应用热图颜色映射，将归一化后的PCA图像转换为伪彩色图像
    heatmap_color = cv2.applyColorMap(comp_img, cv2.COLORMAP_JET)
    #它可以将单通道的灰度图像映射为伪彩色图像，这在可视化某些数据时非常有用，比如深度图、热度图等。cv2.applyColorMap(src, colormap)
    # src：输入的单通道灰度图像。colormap：伪彩色映射的类型，是一个整数值，代表不同的伪彩色映射方法。


    # 调整大小，将伪彩色热图调整大小为原始图像的14倍
    heatmap_color = cv2.resize(heatmap_color, (w*14,h*14))

    return heatmap_color
     

if __name__ == "__main__":
    from dinov2.dinov2.models import build_model_from_cfg
    from easydict import EasyDict as edict
    from dinov2.dinov2.utils.config import get_cfg
    from dinov2.dinov2.utils.utils import load_pretrained_weights
    DEVICE = "cuda"

    #调用get_cfg函数，从指定的配置文件中获取配置参数。
    cfg = get_cfg("dinov2/dinov2/configs/eval/vits14_pretrain.yaml")

    # Model=  arch: vit_small；patch_size: 14；_=only_teacher;embed_dim=imgsize=518
    model, _, embed_dim = build_model_from_cfg(cfg, only_teacher=False)

    #加载预训练的模型权重。
    load_pretrained_weights(model, 'weights/dinov2_vits14.pth', checkpoint_key="teacher")

    model.to(device=DEVICE).eval()#将模型移动到指定的设备上（此处为GPU），并将其设置为评估模式。
    
    pil_image = Image.open("data/demos/LINEMOD.png").convert('RGB')

    #定义一个预处理方法prep
    prep = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    input_tensor = prep(pil_image)[None, ...].cuda()
    #应用预处理管道到图像，并将结果转换为张量，并添加一个额外的维度，以符合模型的输入要求。然后将张量移动到GPU上。


    from sklearn.decomposition import PCA
    # dict_keys(['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks'])

    #使用模型对输入图像进行推断，得到模型的输出。
    out = model(input_tensor,is_training=True )

    #从模型输出中获取描述符（descriptors），并将其从GPU移动到CPU上，并转换为NumPy数组。
    descriptors = out["x_norm_patchtokens"].cpu().detach().numpy()

    #使用PCA对描述符进行降维，这里将维度降到1维。
    pca = PCA(n_components=1).fit(descriptors[0])

    #应用PCA转换到描述符，得到PCA转换后的结果。
    img_pca = pca.transform(descriptors[0])

#调用之前定义的plot_pca函数，将PCA转换后的结果绘制成热图，并将其保存到指定目录下。
    heatmap_color = plot_pca( img_pca.reshape(( 448//14, 448//14 )), save_dir="./")

#将绘制好的热图保存为文件
    cv2.imwrite("data/demos/Linemoddino.png", heatmap_color)

    #用于对一个图像进行特征提取，并对提取的特征进行PCA降维，然后绘制PCA结果的热图，并保存。
