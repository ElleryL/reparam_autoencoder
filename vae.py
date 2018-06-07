import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from data import load_mnist, plot_images, save_images
import matplotlib.pyplot as plt
import torch.distributions as dist


# Load MNIST and Set Up Data
N = 300
D = 784
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
        eps = torch.FloatTensor(log_std.size()).normal_(0, 1)
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

        normal = torch.distributions.Normal(mu, torch.exp(log_std)**2)
        log_q_z_given_x = torch.sum(normal.log_prob(z_tilde))

        return z_tilde,mu,log_std

    def log_q_z_given_x(self,z_tilde,mu,log_std):
        normal = torch.distributions.Normal(mu, torch.exp(log_std)**2)
        log_q_z_given_x = torch.sum(normal.log_prob(z_tilde))
        return log_q_z_given_x

    def prior_log_p_z(self,z):
        normal = dist.Normal(0,1)
        return torch.sum(normal.log_prob(z))


    def bernoulli_decode(self, z_tilde, x):

        h1 = self.Tanh(self.layer21(z_tilde))

        y = self.Sig(self.layer22(h1))

        # for reconstruction we can directly compute p(x|z)
        log_p_x_given_z = F.binary_cross_entropy(y, x, size_average=False)

        return log_p_x_given_z  #normalize


    def forward(self, x):

        z_tilde,mu,log_std = self.encode(x)
        log_p_x_given_z = self.bernoulli_decode(z_tilde,x)

        return log_p_x_given_z,z_tilde,mu,log_std

    def objective_func(self, log_p_x_given_z,mu,log_std):

        KL = -0.5 * torch.mean(torch.mean(1 + log_std - (mu ** 2) - torch.exp(log_std),dim=1))
        neg_elbo = (KL + log_p_x_given_z)

        #neg_elbo = (log_q_z_given_x + self.prior_log_p_z(z_tilde)).mean() - log_p_x_given_z

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

            log_p_x_given_z, z_tilde,mu,log_std= vae(batch_x)

            loss = vae.objective_func(log_p_x_given_z,mu,log_std)

            train_loss += loss
            loss.backward()
            opt.step()
        if i%50 == 0:
            print("Loss at {} is {}".format(i, train_loss/batch_size))
        loss_curve.append(train_loss/batch_size)


    return loss_curve

torch.manual_seed(14)
random.seed(14)
np.random.seed(14)
vae = VAE(D,400,20)
opt = optim.Adam(vae.parameters(), lr=1e-3)

loss_curve = train_vae(vae,opt,2000,50,train_images,train_labels)

plt.plot(loss_curve)
plt.title('Loss Value')
plt.show()