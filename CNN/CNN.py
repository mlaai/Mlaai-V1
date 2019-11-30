import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline 
import numpy as np
from glob import glob

from PIL import Image
import torchvision.transforms as transforms

from torchvision import models
import torch

from torchvision import transforms
transform = transforms.Compose([            
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn as nn
import torch.nn.functional as F
import torch
use_cuda = torch.cuda.is_available()

from matplotlib import pyplot as plt

class CNN(nn.Module):
    """
    Defines methods for Convolutional Neural Network implementations
    """

    
    def faceDetector(img_path):
        """Returns number of faces in an image using OpenCV"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    
    def getDetectorPercent(data, fn):
        """Gets the percent of detected target in image data """
        cnt = 0
        for img in data:
            test = fn(img)
            if test:
                cnt +=1
        return (cnt/(len(data)))*(len(data))  

    def VGG16Predict(img_path):
        """
        Use pre-trained VGG-16 model to obtain index corresponding to 
        predicted ImageNet class for image at specified path
        """
        img = Image.open(img_path)
        convertToTensor = transforms.Compose([transforms.RandomResizedCrop(250),
                                                 transforms.ToTensor()])
        imgTensor = convertToTensor(img)
        imgTensor = imgTensor.unsqueeze(0) 
    
        if torch.cuda.is_available():
            imgTensor = imgTensor.cuda()

        predictedClassIndex = VGG16(imgTensor)
    
        if torch.cuda.is_available():
            predictedClassIndex = predictedClassIndex.cpu()
   
        return predictedClassIndex.data.numpy().argmax()

    def dog_detector(img_path):
        """Returns dog breed index in an image using VGG16Predict"""
        index = VGG16Predict(img_path)
        return (151 <= index and index <= 268)


    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm2d1 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 133)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.norm2d1(self.conv3(x))))

        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
        """returns trained model"""
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf 
    
        for epoch in range(1, n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
        
            ###################
            # train the model #
            ###################
            model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            ######################    
            # validate the model #
            ######################
            model.eval()
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                output = model(data)
                loss = criterion(output, target)
                valid_loss = valid_loss + ((1/ (batch_idx + 1)) * (loss.data - valid_loss))
            
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))
        
            ## save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(model.state_dict(), 'model_scratch.pt')
                valid_loss_min = valid_loss    
        # return trained model
        return model


    def test(loaders, model, criterion, use_cuda):

        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
            
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))


        def predict_breed_transfer(img_path):
            global model_transfer
            global transform_pipeline
            # load the image and return the predicted breed

            image = Image.open(img_path).convert('RGB')

            transform_pipeline = transforms.Compose([transforms.RandomResizedCrop(224),
                                                     transforms.ToTensor()])
            image = transform_pipeline(image)[:3,:,:].unsqueeze(0)
    
            if use_cuda:
                model_transfer = model_transfer.cuda()
                image = image.cuda()
    
            model_transfer.eval()
            idx = torch.argmax(model_transfer(image))
            return class_names[idx]

        def run_app(img_path):
            ## handle cases for a human face, dog, and neither
            breed = predict_breed_transfer(img_path) 
    
            img = cv2.imread(img_path)
            cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.xlabel(img_path, fontsize=16)
            plt.imshow(cv_rgb)
            plt.show()
    
            if dog_detector(img_path):
                print("Dog breed is: " + str(breed))
            elif face_detector(img_path):
                print("Human resembing dog breed " + str(breed))
            else:
                print("Error - Neither dog nor human detected!")
        