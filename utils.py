import torch


def cv2ten(img, device):
    img = (img[:, :, ::-1].transpose(2, 0, 1) / 255. - 0.5) / 0.5
    img_ten = torch.from_numpy(img).float().unsqueeze(0).to(device)
    return img_ten


def ten2cv(img_ten, bgr=True):
    img = img_ten.squeeze(0).mul_(0.5).add_(0.5).mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if bgr:
        img = img[:, :, ::-1]
    return img

