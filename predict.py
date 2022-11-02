import argparse
from PIL import Image
import json
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint', help='Given checkpoint of a network')
parser.add_argument('--top_k', help='Return top k most likely classes')
parser.add_argument('--category_names', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', help='Use GPU for inference', action='store_true')


args = parser.parse_args()

top_k = 1 if args.top_k is None else int(args.top_k)
category_names = "cat_to_name.json" if args.category_names is None else args.category_names
gpu = False if args.gpu is None else True

def build_network(architecture, hidden_units):
    print("Building network ... architecture: {}, hidden_units: {}".format(architecture, hidden_units))
    
    if architecture =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif architecture =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216
        
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(
          nn.Linear(25088, 512),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(256, 102),
          nn.LogSoftmax(dim = 1)
        )

    model.classifier = classifier
    
    return model
# Loading checkpoint and rebuilding the model
def load_model(filepath):

    checkpoint = torch.load(filepath)
    model = build_network(checkpoint['architecture'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model


model = load_model(args.checkpoint)
print(model)

def predict(processed_image, model, topk): 
    model.eval()
    with torch.no_grad():
        logps = model.forward(processed_image.unsqueeze(0))
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return probs.numpy()[0], classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    image_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
    
    return image_transform(image)


probs, predict_classes = predict(process_image(args.image_path), model, top_k)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

classes = []
    
for predict_class in predict_classes:
    classes.append(cat_to_name[predict_class])

print(probs)
print(classes)