from torchvision import transforms
from torchvision.transforms import *
from PIL import Image 
import ipdb
import numpy as np
import os
test_img = '/raid/shchen/videoOPT-Three/datasets/cc3m/validation/2919642852'
test_img = Image.open(test_img)
print(test_img.size)
os.makedirs('./output/augimg',exist_ok=True)

#ipdb.set_trace()

test_img.save('./output/augimg/raw.png')
#trans = transforms.Compose([ToTensor()])
trans1 = transforms.Compose([RandomResizedCrop(224)])
trans2 = transforms.Compose([Resize(224),RandomCrop(224)])
for i in range(10):
    test_img_cropped = trans1(test_img)
    test_img_cropped.save(f'./output/augimg/cropImg_{i}.png')





