import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests
import torch
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim

class DCGAN(object):
    """description of class"""
    def getDataloader(batchSize, imageSize, dataDir):
        """Batch neural network data using dataloader """
        imageAug = transform.Compose([transforms.Resize(imageSize),
                                      transforms.CenterCrop(imageSize),
                                      transforms.ToTensor()])

        imagenetData = datasets.ImageFolder(dataDir,transforms=imageAug)

        dataLoader = torch.utils.data.DataLoader(imagenetData,
                                                 batchSize=batchSize+1,
                                                 shuffle=True)
        return dataLoader

    def imshow(img):
        """Image Show"""
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def scale(x, feature_range=(-1, 1)):
        """Scales an image """
        min, max = feature_range
        x = x * (max - min) + min
        return x            

    def weights_init_normal(m):
        """
        Applies initial weights to certain layers in a model.
        This should initialize only convolutional and linear layers
        Initialize the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
        The bias terms, if they exist, may be left alone or set to 0.
        """
        classname = m.__class__.__name__
    
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    def build_network(d_conv_dim, g_conv_dim, z_size):
        """
        Define model's hyperparameters and instantiate the discriminator and generator 
        """
        D = Discriminator(d_conv_dim)
        G = Generator(z_size=z_size, conv_dim=g_conv_dim)

        D.apply(weights_init_normal)
        G.apply(weights_init_normal)

        print(D)
        print()
        print(G)
    
        return D, G

    def real_loss(D_out):
        """
        Calculates how close discriminator outputs are to being real
        """
        batch_size = D_out.size(0)
        labels = torch.ones(batch_size) * np.random.uniform(0.7, 1.2)
        if train_on_gpu:
            labels = labels.cuda()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels) 
        return loss

    def fake_loss(D_out):
        """
        Calculates how close discriminator outputs are to being fake.
        """
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size) * np.random.uniform(0.0, 0.3) # fake labels = 0.3
        if train_on_gpu:
            labels = labels.cuda()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels) 
        return loss

    def optimizers():
        """
        Optimizers for Discriminator and Generator
        """
        d_optimizer =  optim.Adam(D.parameters(), 0.0005, [0.1, 0.99])
        g_optimizer = optim.Adam(G.parameters(), 0.0005, [0.1, 0.99])
        
        return d_optimizer, g_optimizer

    def train(D, G, n_epochs, absoluteFileName, print_every=50):
        """
        Trains adversarial networks for some number of epochs
        """
    
        # move models to GPU
        if train_on_gpu:
            D.cuda()
            G.cuda()

        # keep track of loss and generated, "fake" samples
        samples = []
        losses = []

        # Get some fixed data for sampling. These are images that are held
        # constant throughout training, and allow us to inspect the model's performance
        sample_size=16
        fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
        fixed_z = torch.from_numpy(fixed_z).float()
        # move z to GPU if available
        if train_on_gpu:
            fixed_z = fixed_z.cuda()

        # epoch training loop
        for epoch in range(n_epochs):

            # batch training loop
            for batch_i, (real_images, _) in enumerate(celeba_train_loader):

                batch_size = real_images.size(0)
                real_images = scale(real_images)

                d_optimizer.zero_grad()
            
                if train_on_gpu:
                    real_images = real_images.cuda()
            
                d_out = D(real_images)
                d_real_loss = real_loss(d_out)
            
                z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                z = torch.from_numpy(z).float()
            
                if train_on_gpu:
                    z = z.cuda()
                
                fake_images = G(z)
            
                d_fake_out = D(fake_images)
                d_fake_loss = fake_loss(d_fake_out)

                d_loss = d_real_loss + d_fake_loss
            
                d_loss.backward()
                d_optimizer.step()

                # 2. Train the generator with an adversarial loss
                g_optimizer.zero_grad()
            
                z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                z = torch.from_numpy(z).float()
            
                if train_on_gpu:
                    z = z.cuda()
                
                fake_images = G(z)
                g_fake_out = D(fake_images)
                g_loss = real_loss(g_fake_out) # use real loss to flip labels
                g_loss.backward()
                g_optimizer.step()       
            
                if batch_i % print_every == 0:
                    # append discriminator loss and generator loss
                    losses.append((d_loss.item(), g_loss.item()))
                    # print discriminator and generator loss
                    print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                            epoch+1, n_epochs, d_loss.item(), g_loss.item()))

            G.eval() 
            samples_z = G(fixed_z)
            samples.append(samples_z)
            G.train() 

        # Save training generator samples
        with open(absoluteFileName, 'wb') as f:
            pkl.dump(samples, f)

        return losses