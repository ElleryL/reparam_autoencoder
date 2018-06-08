import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
from data import load_mnist, plot_images, save_images
import matplotlib.pyplot as plt
import torch.distributions as dist



# Load MNIST and Set Up Data
N = 500
D = 784
S = 50 # sample size for variational params
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = torch.from_numpy(np.round(train_images[0:N])).float()
train_labels = torch.from_numpy(train_labels[0:N]).float()
test_images = torch.from_numpy(np.round(test_images[0:10000])).float()
test_labels = torch.from_numpy(np.round(test_labels[0:10000])).float()



class VAE(nn.Module):
    def __init__(self,input_Dim,hidden_unit,latent_Dim):
        super(VAE,self).__init__()

        # See paper appendix C1

        # Map to latent representative space
        self.layer11 = nn.Linear(input_Dim,hidden_unit)
        self.mu_layer12 = nn.Linear(hidden_unit,latent_Dim)
        self.log_std_layer12 = nn.Linear(hidden_unit,latent_Dim) # now we are at latent space q(z|x)

        # Reconstruction from latent for Bernoull Case
        self.layer21 = nn.Linear(latent_Dim, hidden_unit)
        self.layer22 = nn.Linear(hidden_unit, input_Dim)  # now we have simulated data p(x|z)

        self.Tanh = F.tanh  # activation function for layer1
        self.Sig = F.sigmoid # activation function for output layer for bernoulli case; map to prob

    def reparameterization(self, mu, log_std):
        '''
        sample epi from N(0,1)
        apply smooth function q_z_given_x = g(mu,log_std,epi)

        see equation(10) in the paper

        :param mu: mu from encoding
        :param log_std: log_std from encoding
        :return:
        '''
        #eps = torch.FloatTensor(log_std.size()).normal_(0, 1)
        eps = torch.ones([S,log_std.size()[0],log_std.size()[1]]).normal_(0, 1)
        std = torch.exp(log_std)
        z_tilde = mu + std*eps
        return z_tilde

    def encode(self,x):

        #encode
        h1 = self.Tanh(self.layer11(x))
        mu = (self.mu_layer12(h1))
        log_std = (self.log_std_layer12(h1))

        # Instead of directly sample variational parameters from this latent space q(z|x)
        # reparameterization it first
        z_tilde = self.reparameterization(mu, log_std)

        return z_tilde,mu,log_std

    def log_Q_z_given_x(self,z_tilde,mu,log_std):
        normal = torch.distributions.Normal(mu, torch.exp(log_std)**2)
        log_q_z_given_x = torch.mean(torch.sum(normal.log_prob(z_tilde),dim=2),dim=1)


        return log_q_z_given_x

    def prior_log_p_z(self,z):
        normal = dist.Normal(0,1)
        return torch.mean(normal.log_prob(z.mean(0)),dim=1)


    def bernoulli_decode(self, z_tilde):
        h1 = self.Tanh(self.layer21(z_tilde))
        y = self.Sig(self.layer22(h1))
        return y

    def log_P_x_given_z(self,y,x):
        return F.binary_cross_entropy(y,x,size_average=False)

    def forward(self, x):

        z_tilde,mu,log_std = self.encode(x)
        y = self.bernoulli_decode(z_tilde.mean(0))

        return y,z_tilde,mu,log_std

    def objective_func(self, z_tilde,log_p_x_given_z,mu,log_std):

        KL = -0.5 * torch.sum(1 + log_std - (mu ** 2) - torch.exp(log_std))
        neg_elbo = (KL + log_p_x_given_z)

        log_q_z_given_x = self.log_Q_z_given_x(z_tilde,mu,log_std)

        neg_elbo = ((log_p_x_given_z + self.prior_log_p_z(z_tilde).mean()) - log_q_z_given_x.mean())


        return neg_elbo


def train_vae(vae,opt,iters,batch_size,dataX,dataY):
    permutation = torch.randperm(train_images.size()[0])

    loss_curve = []
    for i in range(iters):
        train_loss = 0
        for m in range(0,dataX.size()[0],batch_size):
            opt.zero_grad()
            indices = permutation[m:m + batch_size]
            batch_x, batch_y = dataX[indices], dataY[indices]

            y, z_tilde,mu,log_std= vae(batch_x)
            log_p_x_given_z = vae.log_P_x_given_z(y,batch_x)
            loss = vae.objective_func(z_tilde,log_p_x_given_z,mu,log_std)

            train_loss += loss
            loss.backward()
            opt.step()
        if i%50 == 0:
            print("Loss at {} is {}".format(i, train_loss/batch_size))
        loss_curve.append(train_loss/batch_size)
    plt.plot(loss_curve)
    plt.title('Loss Value')
    plt.show()
    return loss_curve

def simulateImage(dataX,vae):
    '''
    Checked the decoded image to see if it similar to input image
    :param dataX: an image to be decode and encode
    :param vae: a well trained VAE
    :return:
    '''
    y, z_tilde,mu,log_std= vae(dataX)

    img = np.concatenate((dataX.data.numpy(),y.data.numpy()),axis=0)
    plot_images(img,plt,ims_per_row=10)
    plt.show()



torch.manual_seed(14)
random.seed(14)
np.random.seed(14)
vae = VAE(D,400,20)
opt = optim.Adam(vae.parameters(), lr=1e-3)

loss_curve = train_vae(vae,opt,200,50,train_images,train_labels)

# we do some simple Turing test here
simulateImage(test_images[:10],vae)



