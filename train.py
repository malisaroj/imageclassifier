import argparse
import torch
from torchvision import transforms, datasets, models
from torch import nn
from torch import optim
from collections import OrderedDict


parser = argparse.ArgumentParser(description='Training a neural network on a given dataset')
parser.add_argument('data_dir', help='Path to dataset on which the neural network should be trained on')
parser.add_argument('--save_dir', help='Path to directory where the checkpoint should be saved')
parser.add_argument('--arch', help='Network architecture (default \'vgg16\')')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Number of hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')


args = parser.parse_args()

save_dir = '' if args.save_dir is None else args.save_dir
network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.0025 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True



#defining function to load datas
def load_data(path):
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    #data_transforms =

    # TODO: Load the datasets with ImageFolder

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(train_dir, transform=test_transforms)

    #image_datasets = 

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_dataset, train_dataloader, valid_dataloader, test_dataloader

def build_network(architecture, hidden_units):
    print("Building network ... architecture: {}, hidden_units: {}".format(architecture, hidden_units))
    
    if architecture =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif architecture =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
    elif architecture =='densenet':
        model = models.densenet121(pretrained = True)
        input_units = 1024
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_units, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    print("Finished building network.")
    
    return model
    
    
def train_network(model, epochs, learning_rate, trainloader, validloader, gpu):
    print("Training network ... epochs: {}, learning_rate: {}, gpu used for training: {}".format(epochs, learning_rate, gpu))
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    model.to(device)
    
    
    # Training the network
    steps = 0
    print_every = 10
    train_loss = 0

    # Note: I looked at the notebooks from the last module and decided to do it in a similar way
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate validation accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {train_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(validloader):.3f}, "
                      f"Valid accuracy: {valid_accuracy/len(validloader):.3f}")

                train_loss = 0

                model.train()

    print("Finished training network.")            
    
    return model, criterion

def evaluate_model(model, testloader, criterion, gpu):
    # TODO: Do validation on the test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Validation on the test set
    test_loss = 0
    test_accuracy = 0
    model.eval() # We just want to evaluate and not train the model
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy of test set
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(test_dataloader):.3f}, "
          f"Test accuracy: {test_accuracy/len(test_dataloader):.3f}")
    running_loss = 0
    

def save_model(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    # Saving the checkpoint

    checkpoint = {
        'epochs': epochs,
        'architecture': architecture,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }
    
    checkpoint_path = save_dir + "checkpoint.pth"

    torch.save(checkpoint, 'checkpoint.pth')
    
train_dataset, train_dataloader, valid_dataloader, test_dataloader = load_data(args.data_dir)


model = build_network(network_architecture, hidden_units)
model.class_to_idx = train_dataset.class_to_idx

model, criterion = train_network(model, epochs, learning_rate, train_dataloader, valid_dataloader, gpu)
evaluate_model(model, test_dataloader, criterion, gpu)
save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)  
   

