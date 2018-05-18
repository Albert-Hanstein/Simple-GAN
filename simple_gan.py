#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:55:04 2018

@author: hans
"""
# Import the libraries required
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Set the data parameters
data_mean = 4
data_stddev = 1.25

# Model parameters
g_input_size = 1
g_hidden_size = 50
g_output_size = 1
d_input_size = 100
d_hidden_size = 50
d_output_size = 1
minibatch_size = d_input_size

d_learning_rate = 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999) # For Adam optimisation
num_epochs = 30000
print_interval = 50
d_steps = 1 # This is 'k' in Ian Goodfellow's paper on GANs
g_steps = 1

# Generating real data and generator input data for models
# Real data for the discriminator
def real_distribution_sampler(mu, sigma):
    # This function returns values corresponding to a Gaussian distribution
    # with mean mu and standard deviation sigma.
    return lambda n: torch.Tensor(np.random.normal(mu,sigma, (1,n))) #Gaussian
# Uniform distribution data input into generator
def generator_input_sampler():
    return lambda m,n: torch.rand(m, n)

"""Defining the weights_init function that takes as input
a neural network m and that will initialize all its weights."""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('elu') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('sigmoid') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Models
# Generator
class Generator(nn.Module):
    # Define the layers and number of nodes
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
    
    # Define the feedforward function    
    def forward(self, x):
        x = F.elu(self.map1(x)) #Exponential Linear Unit as activation function
        x = F.sigmoid(self.map2(x)) # for non-linearity
        return self.map3(x)
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))
    
# Functions to extract distribution information from outputs
def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

# Now that we have defined everything, we instantiate
# Instantiate input values
d_sampler = real_distribution_sampler(data_mean, data_stddev)
g_sampler = generator_input_sampler()

# Instantiate the models
G = Generator(input_size = g_input_size, hidden_size = g_hidden_size, output_size = g_output_size)
D = Discriminator(input_size = d_input_size, hidden_size = d_hidden_size, output_size = d_output_size)

# Initialise the weights
G.apply(weights_init)
D.apply(weights_init)

# Calculating error
criterion = nn.BCELoss() # Binary cross entropy

# Optimisers (to adjust weights and decrease error)
d_optimiser = optim.Adam(D.parameters(), lr = d_learning_rate, betas = optim_betas)
g_optimiser = optim.Adam(G.parameters(), lr = g_learning_rate, betas = optim_betas)

# Now we connect them all and train the models
for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D
        D.zero_grad() # Pytorch tends to accumulate gradients, hence use this line.
        
        # 1.1. Train D on real data
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(d_real_data) # Output of discriminator
        """ Target is set to 1 because ideally we want the discriminator to output
            1 for real data. To calculate error, the decision output of D and an
            array of ones is used."""
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))
        d_real_error.backward() # This only computes/stores gradients. Parameters
                                # are not changed yet.
        
        # 1.2. Train D on fake data
        d_gen_input = Variable(g_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach() # Use detach to avoid training G on these labels
        d_fake_decision = D(d_fake_data.t())
        """ Likewise, for this case, the target is zero. """
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))
        d_fake_error.backward()
        d_optimiser.step() # Using the gradients stored from backward() to update parameters
        
    for g_index in range(g_steps):
        # 2. Train G on D's feedback
        G.zero_grad()
        
        gen_input = Variable(g_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data.t())
        """ Calculate error using D's response and 1, because we want the generator to
            output values which will make the discriminator output 1"""
        g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))
        g_error.backward()
        g_optimiser.step() # Update the parameters of G
        
    if epoch % print_interval == 0:
        print("Epoch: %s, D(real/fake)error: %s/%s, G error: %s" % (epoch,
                                                                    extract(d_real_error)[0],
                                                                    extract(d_fake_error)[0],
                                                                    extract(g_error)[0]))
        print("Real stats: %s, Fake stats: %s" % (stats(extract(d_real_data)),
                                                  stats(extract(d_fake_data))))
