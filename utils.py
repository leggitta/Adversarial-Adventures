import glob
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import make_grid


img_path = '/home/alan/Projects/gen_dnd_art/filtered_images/im128/*pkl'
img_files = glob.glob(img_path)

# determine class names
labels = np.array([i.split('/')[-1].split('_')[:3] for i in img_files])
species = np.unique(labels[:, 0]).tolist()
classes = np.unique(labels[:, 1]).tolist()
genders = np.unique(labels[:, 2]).tolist()

class ImSet(Dataset):
    def __init__(self, img_path=img_path):
        super().__init__()
        self.img_files = glob.glob(img_path)
        self.transform = T.Compose([
            T.ToTensor(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.1),
            T.RandomHorizontalFlip(),
            # add random noise and clip
            lambda x: torch.clip(torch.randn(x.shape) / 20 + x, 0, 1),
            T.Normalize(0.5, 0.5)
        ])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, i):
        img_file = self.img_files[i]
        
        # load image
        with open(img_file, 'rb') as fid:
            img = pickle.load(fid)
        
        # apply transforms
        img = self.transform(img).float()
        
        # extract class label
        img_fname = img_file.split('/')[-1]
        species_, class_, gender_, _, _ = img_fname.split('_')
        species_ = species.index(species_)
        class_ = classes.index(class_)
        gender_ = genders.index(gender_)
        
        return (img_fname, img, species_, class_, gender_)

class VariationalEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_size=4096):
        super().__init__()
        
        self.latent_size = latent_size
        
        self.net = nn.Sequential(
            # 128 -> 63
            nn.Conv2d(input_channels, 16, 4, 2),
            nn.LeakyReLU(0.2),
            
            # 63 -> 31
            nn.Conv2d(16, 32, 3, 2),
            nn.LeakyReLU(0.2),
            
            # 31 -> 15
            nn.Conv2d(32, 64, 3, 2),
            nn.LeakyReLU(0.2),

            # 15 -> 7
            nn.Conv2d(64, 128, 3, 2),
            nn.LeakyReLU(0.2),

            # 7 -> 5
            nn.Conv2d(128, 256, 3, 1),
            nn.LeakyReLU(0.2),

            # 5 -> 4
            nn.Conv2d(256, 512, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 4 -> 3
            nn.Conv2d(512, 1024, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 3 -> 2
            nn.Conv2d(1024, 2048, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 2 -> 1
            nn.Conv2d(2048, latent_size, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(latent_size, latent_size),
            # nn.Dropout(0.4)
        )
        
        # parameters for variational autoencoder
        self.mu = nn.Linear(latent_size, latent_size)
        self.sigma = nn.Linear(latent_size, latent_size)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = self.net(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        x = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        return x

class ConditionalEncoder(VariationalEncoder):
    def __init__(self, latent_size=4096):
        super().__init__(input_channels=4, latent_size=latent_size)
        
        self.emb_species = nn.Embedding(len(species), 128**2 // 3 + 128**2 % 3)
        self.emb_class = nn.Embedding(len(classes), 128**2 // 3)
        self.emb_gender = nn.Embedding(len(genders), 128**2 // 3)
        self.emb_reshape = nn.Unflatten(1, (1, 128, 128))


    def forward(self, img, species_, class_, gender_):
        x = self.emb_species(species_)
        y = self.emb_class(class_)
        z = self.emb_gender(gender_)
        
        x = torch.concat([x, y, z], dim=1)
        x = self.emb_reshape(x)
        
        x = torch.concat([img, x], dim=1)
        x = self.net(x)
        
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        x = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return x

    
class Decoder(nn.Module):
    def __init__(self, latent_size=4096):
        super().__init__()
        self.latent_size = latent_size
        self.net = nn.Sequential(
            
            nn.Linear(latent_size, latent_size),
            # nn.Dropout(0.4),
            
            nn.Unflatten(1, (latent_size, 1, 1)),
            
            # 1 -> 2
            nn.ConvTranspose2d(latent_size, 2048, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 2 -> 3
            nn.ConvTranspose2d(2048, 1024, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 3 -> 4
            nn.ConvTranspose2d(1024, 512, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 4 -> 5
            nn.ConvTranspose2d(512, 256, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 5 -> 7
            nn.ConvTranspose2d(256, 128, 3, 1),
            nn.LeakyReLU(0.2),
            
            # 7 -> 15
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.LeakyReLU(0.2),
            
            # 15 -> 31
            nn.ConvTranspose2d(64, 32, 3, 2),
            nn.LeakyReLU(0.2),
            
            # 31 -> 63
            nn.ConvTranspose2d(32, 16, 3, 2),
            nn.LeakyReLU(0.2),

            # 63 -> 128
            nn.ConvTranspose2d(16, 3, 4, 2),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)


class ConditionalDecoder(Decoder):
    def __init__(self, latent_size=4096):
        super().__init__(latent_size)
        
        self.emb_species = nn.Embedding(len(species), latent_size // 3 + latent_size % 3)
        self.emb_class = nn.Embedding(len(classes), latent_size // 3)
        self.emb_gender = nn.Embedding(len(genders), latent_size // 3)
        self.label_net = nn.Linear(2*latent_size, latent_size)
        
    def forward(self, Z, species_, class_, gender_):
        x = self.emb_species(species_)
        y = self.emb_class(class_)
        z = self.emb_gender(gender_)
        
        x = torch.concat([Z, x, y, z], dim=1)
        x = self.label_net(x)
        x = self.net(x)
        return x

    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_size=4096):
        super().__init__()
        self.latent_size = latent_size
        self.enc = VariationalEncoder(latent_size)
        self.dec = Decoder(latent_size)
    
    def forward(self, x):
        return self.dec(self.enc(x))

class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, latent_size=4096):
        super().__init__()
        self.latent_size = latent_size
        self.enc = ConditionalEncoder(latent_size)
        self.dec = ConditionalDecoder(latent_size)
    
    def forward(self, img, species_, class_, gender_):
        Z = self.enc(img, species_, class_, gender_)
        x = self.dec(Z, species_, class_, gender_)
        return x

class ConditionalDiscriminator(ConditionalEncoder):
    def __init__(self, latent_size=4096):
        super().__init__(latent_size)
        self.dsc = nn.Sequential(
            nn.Linear(latent_size, 1),
            nn.Sigmoid()
        )
    def forward(self, img, species_, class_, gender_):
        x = self.emb_species(species_)
        y = self.emb_class(class_)
        z = self.emb_gender(gender_)
        
        x = torch.concat([x, y, z], dim=1)
        x = self.emb_reshape(x)
        
        x = torch.concat([img, x], dim=1)
        x = self.net(x)
        x = self.dsc(x)
        return x
    
def show_tensor(Z, ax, **kwargs):
    if len(Z.shape) > 3:
        Z = Z[0]
    
    if Z.min() < 1:
        Z = (Z + 1) / 2
    
    Z = np.transpose(Z.detach().cpu().numpy(), (1, 2, 0))
    ax.imshow(Z, **kwargs)
    return ax