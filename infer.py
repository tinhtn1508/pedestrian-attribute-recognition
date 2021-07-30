
import torch
import model
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='/content/tinhtn/MyDrive/result/model/pa100k/random_model_best_pth.tar')
parser.add_argument('--image', type=str)
args = parser.parse_args()


_model = model.TopBDNet(num_classes=26, neck=True, double_bottleneck=True, drop_bottleneck_features=True)
checkpoint = torch.load(args.model, map_location='cpu')
_model.load_state_dict(checkpoint['state_dict'])
_model.eval()

train_dataset, val_dataset, num_classes, attr_name, loss_weight = utils.GetDataset('./dataset/pa100k', './dataset/pa100k/pa100k_description.pkl')



tf = transforms.Compose([
                transforms.Resize(size=(256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

img_path = args.image
img = Image.open(img_path).convert('RGB')
img = tf(img)
img.unsqueeze_(0)
conv_output = _model.forward(img)
predict = conv_output[0]
for output in conv_output:
    predict = torch.max(predict, output)

prob = torch.sigmoid(predict).detach().numpy()[0]
output = np.where(prob > 0.5, 1, 0)
labels = train_dataset.toImageAttribute(output)

prob_labels = []
for p, o in zip(prob, output):
    if o == 1:
        prob_labels.append(p)

res = {labels[i]: int(prob_labels[i]*100) for i in range(len(prob_labels))}

img = mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.show()
print(res)