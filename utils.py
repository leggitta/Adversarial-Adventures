import cv2
import glob
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
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

keypoint_names = {
    0: 'nose',
    1: 'right eye',
    2: 'left eye',
    3: 'right ear',
    4: 'left ear',
    5: 'right shoulder',
    6: 'left shoulder',
    7: 'right elbow',
    8: 'left elbow',
    9: 'right hand',
    10: 'left hand',
    11: 'right hip',
    12: 'left hip',
    13: 'right knee',
    14: 'left knee',
    15: 'right foot',
    16: 'left foot'
}


# left/right defined from view perspective
edges = [
    (0, 1),    # nose to right eye
    (0, 2),    # nose to left eye
    (2, 4),    # left eye to left ear
    (1, 3),    # right eye to right ear
    (6, 8),    # left shoulder to left elblow
    (8, 10),   # left elbow to left hand
    (5, 7),    # right shoulder to right elbow
    (7, 9),    # right elbow to right hand
    (5, 11),   # right shoulder to right hip
    (11, 13),  # right hip to right knee
    (13, 15),  # right knee to right foot
    (6, 12),   # left shoulder to left hip
    (12, 14),  # left hip to left knee
    (14, 16),  # left knee to left foot
    (5, 6),    # right shoulder to left shoulder
    (11, 12),  # right hip to left hip
]

class Pose:
    '''
    
    A pose detected from an Image
    Contains Image information, 17 keypoints, a bounding box
    
    
    Attributes
    - Z (Image data tensor)
    - keypoints
    - boxes
    
    
    Methods
    - plot 
    - resize
    
    '''
    def __init__(self, index=0, **kwargs):
        for k, v in kwargs.items():
            if type(v) == torch.Tensor:
                v = v.cpu().numpy()[index]
            setattr(self, k, v)
        self.dict = kwargs
        self.box = self.boxes
        self.Z = cv2.imread(self.filename)[:, :, ::-1]
        
    def plot(self, ax=None, show=False):
        if ax is None:
            fig, ax = plt.subplots()
        
        # plot image
        ax.imshow(self.Z)
        
        # plot bounding box
        x0, y0, x1, y1 = self.box
        rect = Rectangle((x0, y0), x1-x0, y1-y0, alpha=0.1)
        ax.add_patch(rect)
        
        # plot keypoints
        x = self.keypoints[:, 0]
        y = self.keypoints[:, 1]
        for (i, j) in edges:
            ax.plot([x[i], x[j]], [y[i], y[j]], 'ro-')
            
        if show:
            plt.show()
        
    def resize(self, w=128, h=128):
        # transform bounding box and keypoints
        H, W, _ = self.Z.shape
        
        new_box = []
        x0, y0, x1, y1 = self.box
        
        # compute box for cropping image
        w_p = x1 - x0
        h_p = y1 - y0
        
        i0 = max(int(x0 - w_p/4), 0)
        i1 = min(int(x1 + w_p/4), W)
        
        j0 = max(int(y0 - h_p/4), 0)
        j1 = min(int(y1 + h_p/4), H)
        
        # rescale bounding box coordinates
        x0 = (x0-i0) / (i1-i0) * w
        y0 = (y0-j0) / (j1-j0) * h
        x1 = (x1-i0) / (i1-i0) * w
        y1 = (y1-j0) / (j1-j0) * h
        self.box = np.r_[x0, y0, x1, y1]
        
        self.keypoints[:, 0] = (self.keypoints[:, 0] - i0) / (i1-i0) * w
        self.keypoints[:, 1] = (self.keypoints[:, 1] - j0) / (j1-j0) * h

        # resize image
        self.Z = cv2.resize(self.Z[j0:j1, i0:i1], (w, h))
        self.box

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