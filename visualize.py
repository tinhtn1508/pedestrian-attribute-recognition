from PIL import Image
import numpy as np
import torch
import model
import utils
import torchvision.transforms as transforms
import matplotlib.cm as mpl_color_map
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):

        conv_output = self.model.forward(x, return_featuremaps=True)
        conv_output[0].register_hook(self.save_gradient)
        return conv_output[0], conv_output[1:]

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        predict = model_output[0]
        for output in model_output:
            predict = torch.max(predict, output)

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, predict.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads
        self.model.train()
        # Backward pass with specified target
        predict.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients

        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam

if __name__ == "__main__":
    train_dataset, val_dataset, num_classes, attr_name, loss_weight = utils.GetDataset('./dataset/pa100k', './dataset/pa100k/pa100k_description.pkl')
    train_dataset.transform = transforms.Compose([
                transforms.Resize(size=(256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    ids = []
    for i in range(600, 700):
        img, label, path = train_dataset.__getitem__(i)
        l = train_dataset.toImageAttribute(label)
        if "ShortSleeve" in l:
            ids.append(i)



    target = "ShortSleeve"
    method = "random"
    # path = '/home/tinhtn/workspace/personal/pedestrian-attribute-recognition/output/'
    for i in ids:
        _model = model.TopBDNet(num_classes=26,
        neck=True, double_bottleneck=True, drop_bottleneck_features=True)

        checkpoint = torch.load('random_model_best_pth.tar', map_location='cpu')
        _model.load_state_dict(checkpoint['state_dict'])


        grad_cam = GradCam(_model)
        id = i
        img, label, path = train_dataset.__getitem__(id)
        index = []
        for i, value in enumerate(label):
            if value == 1:
                index.append(i)


        print(index)
        print(train_dataset.toImageAttribute(label))
        origin = train_dataset.getOriginImage(id)
        origin = keras.preprocessing.image.img_to_array(origin)

        img.unsqueeze_(0)
        cam = grad_cam.generate_cam(img, 13)

        heatmap = np.uint8(255*cam)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((origin.shape[1], origin.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.5 + origin
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        plt.imshow(superimposed_img)
        plt.savefig('output/' +target+"_"+str(id)+"_"+method+'.png')
