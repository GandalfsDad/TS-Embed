from torch import nn


class Embedder(nn.Module):
    def __init__(self, timesteps, features, embed_dim=32, hidden_dim = 128, dropout=0.5, **kwargs):

        self.__timesteps = timesteps
        self.__features = features
        self.__embed_dim = embed_dim
        self.__hidden_dim = hidden_dim
        self.__dropout = dropout
        
        self.__passkwargs(**kwargs)

        super(Embedder, self).__init__()  

        self.__buildEncoder()
        self.__buildDecoder()   
        
    def __buildEncoder(self):
        mp2c, mp2h, mp2w = self.__flatmp2_dim()

        self.encoder = nn.Sequential(
            # Input shape: (batch_size, 1, t, f)
            nn.Conv2d(1, self.__c1_channels, self.__c1_kernel_size, stride=self.__c1_stride, padding=self.__c1_padding),
            nn.Dropout(self.__dropout),
            nn.ReLU(),
            nn.MaxPool2d(self.__mp1_kernel_size),
            nn.Conv2d(self.__c1_channels, self.__c2_channels, self.__c2_kernel_size, stride=self.__c2_stride, padding=self.__c2_padding),
            nn.Dropout(self.__dropout),
            nn.ReLU(),
            nn.MaxPool2d(self.__mp2_kernel_size),
            nn.Flatten(),
            nn.Linear(mp2c*mp2h*mp2w,self.__hidden_dim),
            nn.Dropout(self.__dropout),
            nn.Linear(self.__hidden_dim, self.__embed_dim),
            # Output shape: (batch_size, embed_dim)
        )

    def __buildDecoder(self):
        mp2c, mp2h, mp2w = self.__flatmp2_dim()
        ct_stride = (self.__c2_stride[0]+1, self.__c2_stride[1])
        ct2_stride = (self.__c1_stride[0]+1, self.__c1_stride[1])

        self.decoder = nn.Sequential(
            # Input shape: (batch_size, embed_dim)
            nn.Linear(self.__embed_dim, self.__hidden_dim),
            nn.Linear(self.__hidden_dim, mp2c*mp2h*mp2w),
            nn.ReLU(),
            nn.Unflatten(1, (mp2c, mp2h, mp2w)),
            nn.ConvTranspose2d(mp2c, self.__c1_channels, self.__c2_kernel_size, stride=ct_stride, padding=self.__c2_padding, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.__c1_channels, 1, self.__c1_kernel_size, stride=ct2_stride, padding=self.__c1_padding, output_padding=(1, 0)),
            nn.ReLU(),
            # Output shape: (batch_size, 1, t, f)
        )

    def __flatmp2_dim(self):
        height = int((self.__timesteps - self.__c1_kernel_size[0] + 2*self.__c1_padding[0]) // self.__c1_stride[0] + 1)
        height = int((height - self.__mp1_kernel_size[0]) // self.__mp1_kernel_size[0] + 1)
        height = int((height - self.__c2_kernel_size[0] + 2*self.__c2_padding[0]) // self.__c2_stride[0] + 1)
        height = int((height - self.__mp2_kernel_size[0]) // self.__mp2_kernel_size[0] + 1)

        width = int((self.__features - self.__c1_kernel_size[1] + 2*self.__c1_padding[1]) // self.__c1_stride[1] + 1)
        width = int((width - self.__mp1_kernel_size[1]) // self.__mp1_kernel_size[1] + 1)
        width = int((width - self.__c2_kernel_size[1] + 2*self.__c2_padding[1]) // self.__c2_stride[1] + 1)
        width = int((width - self.__mp2_kernel_size[1]) // self.__mp2_kernel_size[1] + 1)

        return (self.__c2_channels, height, width)

                    

    def __passkwargs(self, **kwargs):

        if 'c1_channels' in kwargs:
            self.__c1_channels = kwargs['c1_channels']
        else:
            self.__c1_channels = 16

        if 'c1_kernel_size' in kwargs:
            self.__c1_kernel_size = kwargs['c1_kernel_size']
        else:
            self.__c1_kernel_size = (3, 3)

        if 'c1_stride' in kwargs:
            self.__c1_stride = kwargs['c1_stride']
        else:
            self.__c1_stride = (1, 1)
        
        if 'c1_padding' in kwargs:
            self.__c1_padding = kwargs['c1_padding']
        else:
            self.__c1_padding = (1, 1)

        if 'mp1_kernel_size' in kwargs:
            self.__mp1_kernel_size = kwargs['mp1_kernel_size']
        else:
            self.__mp1_kernel_size = (2, 1)
        
        if 'c2_channels' in kwargs:
            self.__c2_channels = kwargs['c2_channels']
        else:
            self.__c2_channels = 32

        if 'c2_kernel_size' in kwargs:
            self.__c2_kernel_size = kwargs['c2_kernel_size']
        else:
            self.__c2_kernel_size = (3, 3)
        
        if 'c2_stride' in kwargs:
            self.__c2_stride = kwargs['c2_stride']
        else:
            self.__c2_stride = (1, 1)
        
        if 'c2_padding' in kwargs:
            self.__c2_padding = kwargs['c2_padding']
        else:
            self.__c2_padding = (1, 1)
        
        if 'mp2_kernel_size' in kwargs:
            self.__mp2_kernel_size = kwargs['mp2_kernel_size']
        else:
            self.__mp2_kernel_size = (2, 1)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)