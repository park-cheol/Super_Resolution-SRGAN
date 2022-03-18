# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

Implementation of SRGAN in pytorch

- 기존의 많은 연구들은 Pixel-Wise Difference(i.e. MSE)들을 이용
- Pixel-Wise Difference를 사용할 경우 PSNR, SSIM은 높은 스코어를 기록하지만 사람이 보는 것과 상당한 불일치를 야기
- 특히 High Frequency(or High Texture detail) 정보의 손실이 심함
- 왜냐하면 **MSE는 Pixel-Wise Average of Possible Solutions**이므로 Blur가 심함

![1](https://user-images.githubusercontent.com/76771847/158945866-72d2b2d6-17a0-4c8b-806e-8d042d42d6d6.png)

**[ Contribution 요약 ]**

- **GAN Loss + Pretrained VGG ConvLayer에서 FeatureMap 추출 후 Loss 구함**
- **Perceptual Loss = Adversarial loss + Content Loss**

# Architecture

![2](https://user-images.githubusercontent.com/76771847/158945949-33ffc901-a0b3-4ea5-b4c8-1809b2ee7c0a.png)

# Loss Function

**1. GAN Loss**

![5](https://user-images.githubusercontent.com/76771847/158945954-5da59cbc-15a3-4cb9-9bef-6f74e21e0d0f.png)

**2. Content Loss**

![4](https://user-images.githubusercontent.com/76771847/158945956-e522394d-9dcf-4741-a315-8e2248b05037.png)

**3. Full Loss**

![3](https://user-images.githubusercontent.com/76771847/158945957-3f3c8541-6ede-4463-afb6-fc7dd67e3e7c.png)

# Implementation

> python main.py --gpu 0 --batch-size ... 

# Reference

[SRGAN paper](https://arxiv.org/abs/1609.04802)

[SRGAN code](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/models.py)

