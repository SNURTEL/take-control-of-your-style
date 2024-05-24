#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1


# In[3]:


transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

dataset_x = ImageFolder(root="data/photo", transform=transform)
dataset_y = ImageFolder(root="data/monet", transform=transform)

train_loader_x = DataLoader(dataset_x, batch_size=batch_size, shuffle=True)
train_loader_y = DataLoader(dataset_y, batch_size=batch_size, shuffle=True)


#

# # Architecture and losses

# In[4]:


# Define the LSGAN loss as in https://arxiv.org/pdf/1611.04076
def lsgan_loss(pred, target):
    return torch.mean((pred - target) ** 2)


# Define the cycle consistency loss as in https://arxiv.org/pdf/1703.10593
def cycle_consistency_loss(real, reconstructed):
    return torch.mean(torch.abs(real - reconstructed))


# In[5]:


import torch
import torch.nn as nn


# U net inspired architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),  # Normalization layer, equivalent to batch norm but for style transfer
            nn.ReLU(True),
            # Downsampling
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            # Residual blocks - pretty deep network - avoid vanishing gradients
            *[ResidualBlock(256) for _ in range(9)],
            # Upsampling
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


# CNN with linear head
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# # Training
# $$
# L(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda \mathcal{L}_{cyc}(G, F)
# $$
# from https://arxiv.org/pdf/1703.10593
# - update the discriminators with \mathcal{L}_{GAN} (adversarial losses)
# - update the generators with \mathcal{L}_{GAN} and \mathcal{L}_{cyc} (adversarial and cycle consistency losses)

# In[6]:


import pickle
from torch import optim
import os

G = Generator().to(device)
F = Generator().to(device)
Dx = Discriminator().to(device)
Dy = Discriminator().to(device)

lr = 0.0002
beta1 = 0.5
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
F_optimizer = optim.Adam(F.parameters(), lr=lr, betas=(beta1, 0.999))
Dx_optimizer = optim.Adam(Dx.parameters(), lr=lr, betas=(beta1, 0.999))
Dy_optimizer = optim.Adam(Dy.parameters(), lr=lr, betas=(beta1, 0.999))


cycle_losses = []
Dx_losses = []
Dy_losses = []
G_losses = []
F_losses = []
os.makedirs("weights", exist_ok=True)


def train(
    epochs,
    train_loader_x,
    train_loader_y,
    G,
    F,
    Dx,
    Dy,
    G_optimizer,
    F_optimizer,
    Dx_optimizer,
    Dy_optimizer,
    cycle_losses,
    Dx_losses,
    Dy_losses,
    G_losses,
    F_losses,
    lambda_param=10,
    weight_dir="weights",
):
    for epoch in range(1, epochs + 1):
        for i, (real_x, real_y) in enumerate(zip(train_loader_x, train_loader_y)):
            real_x = real_x[0]
            real_y = real_y[0]  # drop labels generated by ImageFolder
            real_x = real_x.to(device)
            real_y = real_y.to(device)

            # Train Discriminators
            Dx_optimizer.zero_grad()
            Dy_optimizer.zero_grad()

            fake_y = G(real_x)
            fake_x = F(real_y)

            # adversarial losses of the discriminators min log(1 - D(x)) + log(D(G(x))) => min (1 - D(x))^2 + D(G(x))^2
            Dx_real_loss = lsgan_loss(Dx(real_x), torch.ones_like(Dx(real_x)))
            Dx_fake_loss = lsgan_loss(Dx(fake_x.detach()), torch.zeros_like(Dx(fake_x)))
            Dx_loss = (Dx_real_loss + Dx_fake_loss) * 0.5

            Dy_real_loss = lsgan_loss(Dy(real_y), torch.ones_like(Dy(real_y)))
            Dy_fake_loss = lsgan_loss(Dy(fake_y.detach()), torch.zeros_like(Dy(fake_y)))
            Dy_loss = (Dy_real_loss + Dy_fake_loss) * 0.5

            # backpropagate discriminator losses
            Dx_loss.backward()
            Dy_loss.backward()
            Dx_optimizer.step()
            Dy_optimizer.step()

            # Train Generators
            G_optimizer.zero_grad()
            F_optimizer.zero_grad()

            fake_y = G(real_x)
            fake_x = F(real_y)

            # adversarial losses of the generators max log(D(G(x))) => min (1 - D(G(x)))^2
            G_loss = lsgan_loss(Dy(fake_y), torch.ones_like(Dy(fake_y)))
            F_loss = lsgan_loss(Dx(fake_x), torch.ones_like(Dx(fake_x)))

            cycle_x = F(fake_y)
            cycle_y = G(fake_x)

            # consistency loss - l1 norm between real and reconstructed images - min l1 error
            cycle_x_loss = cycle_consistency_loss(real_x, cycle_x)
            cycle_y_loss = cycle_consistency_loss(real_y, cycle_y)
            total_cycle_loss = (cycle_x_loss + cycle_y_loss) * lambda_param

            # backpropagate generator losses sum adversarial and cycle loss
            total_G_loss = G_loss + total_cycle_loss
            total_F_loss = F_loss + total_cycle_loss

            total_G_loss.backward(
                retain_graph=True
            )  # otherwise cannot backward pass total_F_loss -> reuse total_cycle loss
            total_F_loss.backward()
            G_optimizer.step()
            F_optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader_x)}], "
                    f"cycle loss {total_cycle_loss.item()}"
                    f"Dx Loss: {Dx_loss.item()}, Dy Loss: {Dy_loss.item()}, "
                    f"G Loss: {total_G_loss.item()}, F Loss: {total_F_loss.item()}"
                )

                cycle_losses.append(total_cycle_loss.item())
                Dx_losses.append(Dx_loss.item())
                Dy_losses.append(Dy_loss.item())
                G_losses.append(total_G_loss.item())
                F_losses.append(total_F_loss.item())

        if epoch % 10 == 0:
            torch.save(G.state_dict(), f"{weight_dir}/G_epoch_{epoch}.pth")
            torch.save(F.state_dict(), f"{weight_dir}/F_epoch_{epoch}.pth")
    return cycle_losses, Dx_losses, Dy_losses, G_losses, F_losses


# # Experiments influence of lambda - cycle consistency loss
# lower is expected to encourage more style transfer wheres higher values should encourage more content preservation

# In[ ]:


lambdas = [1, 2, 5, 10, 20]
weight_dir = "weights"
losses_dir = "losses"
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(losses_dir, exist_ok=True)
lr = 0.0002
beta1 = 0.5
num_epochs = 30
for l in lambdas:
    weight_dir_for_lambda = f"{weight_dir}/{l}"
    os.makedirs(weight_dir_for_lambda, exist_ok=True)
    G = Generator().to(device)
    F = Generator().to(device)
    Dx = Discriminator().to(device)
    Dy = Discriminator().to(device)

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    F_optimizer = optim.Adam(F.parameters(), lr=lr, betas=(beta1, 0.999))
    Dx_optimizer = optim.Adam(Dx.parameters(), lr=lr, betas=(beta1, 0.999))
    Dy_optimizer = optim.Adam(Dy.parameters(), lr=lr, betas=(beta1, 0.999))

    cycle_losses, Dx_losses, Dy_losses, G_losses, F_losses = train(
        30,
        train_loader_x,
        train_loader_y,
        G,
        F,
        Dx,
        Dy,
        G_optimizer,
        F_optimizer,
        Dx_optimizer,
        Dy_optimizer,
        cycle_losses,
        Dx_losses,
        Dy_losses,
        G_losses,
        F_losses,
        lambda_param=l,
        weight_dir=weight_dir_for_lambda,
    )
    with open(f"losses_{l}.pkl", "wb") as f:
        pickle.dump(
            {
                "cycle_losses": cycle_losses,
                "Dx_losses": Dx_losses,
                "Dy_losses": Dy_losses,
                "G_losses": G_losses,
                "F_losses": F_losses,
            },
            f,
        )
