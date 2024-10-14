import torch
import numpy
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from pathlib import Path
from PIL import Image
import cv2 
from torchvision import transforms


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
    h,w = pca_image.shape
    comp = pca_image
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))

    comp_img = (comp - comp_min) / (comp_max - comp_min)

    comp_img = (comp_img * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(comp_img, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w*14,h*14))

    cv2.imwrite("headmap.jpg", heatmap_color)


if __name__ == "__main__":
    # 导入所需的模块和函数
    from dinov2.dinov2.models import build_model_from_cfg  # 导入构建模型的函数
    from easydict import EasyDict as edict  # 导入易于使用的字典类型
    from dinov2.dinov2.utils.config import get_cfg  # 导入获取配置的函数
    from dinov2.dinov2.utils.utils import load_pretrained_weights  # 导入加载预训练权重的函数
    from sklearn.decomposition import PCA  # 导入主成分分析模块
    from PIL import Image  # 导入图像处理模块
    import torchvision.transforms as transforms  # 导入图像转换模块

    # 从配置文件中获取配置
    cfg = get_cfg("dinov2/dinov2/configs/eval/vitl14_pretrain.yaml")

    # 构建模型
    model, _, embed_dim = build_model_from_cfg(cfg, only_teacher=False)

    # 加载预训练权重
    load_pretrained_weights(model, 'assets/dinov2_vitl14_pretrain.pth', checkpoint_key="teacher")

    # 将模型设置为评估模式
    model.eval()

    # 使用 PIL 打开图像并进行预处理
    pil_image = Image.open("ppw.png").convert('RGB')  # 打开图像文件并转换为 RGB 格式
    prep = transforms.Compose([  # 创建图像预处理管道
        transforms.Resize((448, 448)),  # 调整图像大小为 448x448
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 归一化处理
    ])
    input_tensor = prep(pil_image)[None, ...]  # 对图像进行预处理并增加一个维度

    # 运行模型推理并获取描述子
    out = model(input_tensor, is_training=True)
    descriptors = out["x_norm_patchtokens"].detach().numpy()

    # 对描述子进行主成分分析
    pca = PCA(n_components=1).fit(descriptors[0])
    img_pca = pca.transform(descriptors[0])

    # 绘制 PCA 结果图
    plot_pca(img_pca.reshape((448 // 14, 448 // 14)), save_dir="./")

