import os

import numpy as np
import torch
from torch.autograd import Variable

import torchvision.transforms as transforms
from PIL import Image
import skimage.io
# from torchvision import transforms as trn

import densenet as models


cwd = os.getcwd()
model_dir = 'models'
densenet = models.densenet161(pretrained=True, model_dir=os.path.join(cwd, model_dir))
densenet.eval()

img_scale = 224
img_crop = 224

# preprocess = transforms.Compose([
#     # trn.Scale(img_scale),
#     # trn.CenterCrop(img_crop),
#     # trn.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image_path = 'resources/cat.jpg'

img = Image.open(image_path)  # (640, 480), RGB
img = transforms.Compose([
            transforms.Scale(img_scale),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            normalize,
        ])(img)

img = img.unsqueeze(0)
# img = img.view([-1, 3, img_crop, img_crop])  # [1, 3, 224, 224]
# img = img.cuda()
input_var = Variable(img, volatile=True)
result = densenet.forward(input_var)


# I = skimage.io.imread(image_path)
# # handle grayscale input images
# if len(I.shape) == 2:
#     I = I[:, :, np.newaxis]
#     I = np.concatenate((I, I, I), axis=2)
#
# I = I.astype('float32') / 255.0
# I = torch.from_numpy(I.transpose([2, 0, 1]))  # [3, 360, 480]
# # print(I.size())
# print(type(I))
# I = Variable(preprocess(I), volatile=True)
# I = I.unsqueeze(0)
# result = densenet.forward(I)


predict = result[0].squeeze()
maxs, indices = torch.max(predict,0)
# print(maxs)
# print(indices)
fc = result[1].squeeze()
att = result[2].squeeze()
att = torch.transpose(att, 0, 2)
print(fc.size())
print(att.size())
print(type(fc))
# conv_feat = conv_feat.cpu().data.numpy()
# conv_feat = np.squeeze(conv_feat)  # (2688, 7, 7)
# conv_feat = conv_feat.transpose()
# conv_feat = conv_feat.reshape(-1, conv_feat.shape[2])
#
# fc_feat = fc_feat.cpu().data.numpy()
# fc_feat = np.squeeze(fc_feat)