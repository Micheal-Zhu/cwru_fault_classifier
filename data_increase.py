import torch
import numpy as np
from torchvision import transforms


class Normalize(object):
    """标准化时间序列数据"""

    def __init__(self, mean=None, std=None):
        """
        Args:
            mean (float): 如果为None，则使用数据的均值
            std (float): 如果为None，则使用数据的标准差
        """
        self.mean = mean
        self.std = std

    def __call__(self, series):
        if self.mean is None or self.std is None:
            # 计算当前序列的均值和标准差
            current_mean = torch.mean(series)
            current_std = torch.std(series)
        else:
            current_mean = self.mean
            current_std = self.std

        # 避免除以0
        if current_std == 0:
            current_std = 1e-10

        return (series - current_mean) / current_std


class RandomNoise(object):
    """添加随机噪声"""

    def __init__(self, noise_level=0.05):
        """
        Args:
            noise_level (float): 噪声级别，相对于数据的标准差
        """
        self.noise_level = noise_level

    def __call__(self, series):
        std = torch.std(series)
        noise = torch.randn_like(series) * std * self.noise_level
        return series + noise


class RandomScale(object):
    """随机缩放时间序列"""

    def __init__(self, scale_range=(0.8, 1.2)):
        """
        Args:
            scale_range (tuple): 缩放范围 (min, max)
        """
        self.scale_range = scale_range

    def __call__(self, series):
        scale_factor = np.random.uniform(*self.scale_range)
        return series * scale_factor


class RandomShift(object):
    """随机平移时间序列"""

    def __init__(self, shift_range=(-0.1, 0.1)):
        """
        Args:
            shift_range (tuple): 平移范围 (min, max)，相对于数据的标准差
        """
        self.shift_range = shift_range

    def __call__(self, series):
        std = torch.std(series)
        shift = np.random.uniform(*self.shift_range) * std
        return series + shift


class RandomStretch(object):
    """随机拉伸或压缩时间序列"""

    def __init__(self, stretch_range=(0.9, 1.1)):
        """
        Args:
            stretch_range (tuple): 拉伸范围 (min, max)
        """
        self.stretch_range = stretch_range

    def __call__(self, series):
        stretch_factor = np.random.uniform(*self.stretch_range)
        old_length = series.shape[0]
        new_length = int(old_length * stretch_factor)

        # 使用线性插值调整大小
        stretched = torch.nn.functional.interpolate(
            series.unsqueeze(0).unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=True
        ).squeeze()

        # 如果拉伸后长度大于原始长度，截取中间部分
        if new_length > old_length:
            start = (new_length - old_length) // 2
            return stretched[start:start + old_length]
        # 如果压缩后长度小于原始长度，填充0
        else:
            result = torch.zeros_like(series)
            start = (old_length - new_length) // 2
            result[start:start + new_length] = stretched
            return result


class ToTensor(object):
    """将numpy数组转换为torch张量"""

    def __call__(self, series):
        return torch.from_numpy(series).float()


def get_train_transform():
    # 组合transform的示例
    train_transform = transforms.Compose([
        Normalize(),
        RandomNoise(noise_level=0.03),
        RandomScale(scale_range=(0.9, 1.1)),
        RandomShift(shift_range=(-0.05, 0.05)),
        # RandomStretch(stretch_range=(0.95, 1.05))  # 可选
    ])

def get_test_transform():
    # 测试时使用的transform
    test_transform = transforms.Compose([
        Normalize(),
    ])
