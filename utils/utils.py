import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid


def sample(niter, img_lr, img_hr, generator):
    generator.eval()
    fake_hr = generator(img_lr)

    img_lr = make_grid(img_lr, nrow=1, normalize=True)
    fake_hr = make_grid(fake_hr, nrow=1, normalize=True)
    img_hr = make_grid(img_hr, nrow=1, normalize=True)
    img_grid = torch.cat((fake_hr, img_hr), -1)

    save_image(img_lr, "output/%d_lr.png" % niter, normalize=False)
    save_image(img_grid, "output/%d_hr.png" % niter, normalize=False)
