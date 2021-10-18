import torch



class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, max_value):
        self.name = "PSNR"
        self.max_value = max_value  # 255 or 1

    @staticmethod
    def __call__(img1, img2, max_value):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(max_value / torch.sqrt(mse))


if __name__ == '__main__':
    i = torch.randn(1, 3, 20, 20)
    j = i / 1.000001
    p = PSNR()
    print(p(i, j).item())
