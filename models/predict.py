import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def predict(image, dir, model_choice = 'resnet34', prob = False):
    """
    This function takes a preprocessed image array as input as returns the predicted label.
    The input can be a single image or a list.
    """
    # set device and transform which will be used further
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])
    # load model save according to model_choice
    if model_choice == 'resnet34':
        model = models.resnet34()
        model.fc = nn.Linear(in_features = 512, out_features = 2, bias = True)
        model = model.to(device)
        model.load_state_dict(torch.load(dir + '/models/resnet34.pth'))
    elif model_choice == 'resnet18':
        model = models.resnet18()
        model.fc = nn.Linear(in_features = 512, out_features = 2, bias = True)
        model = model.to(device)
        model.load_state_dict(torch.load(dir + '/models/resnet18.pth'))
    elif model_choice == 'resnet50':
        model = models.resnext50_32x4d()
        model.fc = nn.Linear(in_features = 2048, out_features = 2, bias = True)
        model = model.to(device)
        model.load_state_dict(torch.load(dir + '/models/resnet50.pth'))
    elif model_choice == 'alexnet':
        model = models.alexnet()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_features, 2)
        model = model.to(device)
        model.load_state_dict(torch.load(dir + '/models/alexnet.pth'))
    elif model_choice == 'mobilenet':
        model = models.mobilenet_v3_large()
        model.classifier[3] = nn.Linear(in_features = 1280, out_features = 2, bias = True)
        model = model.to(device)
        model.load_state_dict(torch.load(dir + '/models/mobilenet.pth'))
    else:
        pass
    
    # check whether the input is a list and predict
    if not isinstance(image, list):
        input = np.stack((image,) * 3, axis = -1)
        input = transform(input).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(input)
        
        res_prob = F.softmax(output, dim = 1).cpu().detach().numpy()[0]
        res_label = torch.max(output, dim = 1)[1].item()

    else:
        input = [np.stack((array,) * 3, axis = -1) for array in image]
        input = [transform(array).unsqueeze(0).to(device) for array in input]

        model.eval()
        with torch.no_grad():
            output = [model(array) for array in input]
        
        res_prob = [F.softmax(array, dim = 1).cpu().detach().numpy()[0] for array in output]
        res_label = [torch.max(array, dim = 1)[1].item() for array in output]
    
    if prob: # return the prob of each label
        return res_prob
    else: # return the predicted label
        return res_label

# this class is used to generate heat-map
class Grad_Cam(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gradient = None
        self.resnet34 = model
        self.features = nn.Sequential(self.resnet34.conv1,
                                      self.resnet34.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size = 3,
                                                   stride = 2,
                                                   padding = 1,
                                                   dilation = 1,
                                                   ceil_mode = False),
                                      self.resnet34.layer1, 
                                      self.resnet34.layer2, 
                                      self.resnet34.layer3, 
                                      self.resnet34.layer4)
        self.avgpool = self.resnet34.avgpool
        self.classifier = self.resnet34.fc

    def activations_hook(self,grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self,x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

def heat_map(image, dir):
    """
    This function takes a preprocessed image array as input as returns the GradCam heat map.
    """
    # set device and transform which will be used further
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])
    # load the best model to generate the heatmap
    model = models.resnet34()
    model.fc = nn.Linear(in_features = 512, out_features = 2, bias = True)
    model = model.to(device)
    model.load_state_dict(torch.load(dir + '/models/resnet34.pth'))
    grad_cam = Grad_Cam(model = model)
    _ = grad_cam.eval()
    # preprocess the input
    image = np.stack((image,) * 3, axis = -1)
    input = transform(image).unsqueeze(0).to(device)
    # make predictions
    output = grad_cam(input)
    values, _ = torch.topk(output, 2)
    index = torch.max(output == values[0][1], dim = 1)[1]
    output[:, index].backward(retain_graph = True)
    # calculate gradients
    gradients = grad_cam.get_gradient()
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])
    activations = grad_cam.get_activations(input).detach()

    for k in range(224):
        activations[:, k, :, :] *= pooled_gradients[k]
    # generate heat map
    heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)