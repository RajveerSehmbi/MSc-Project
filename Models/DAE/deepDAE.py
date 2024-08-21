import sys
sys.path.append('/vol/bitbucket/rs218/Project')

import torch.nn as nn
import torch
import pandas as pd
import variables

input_dim = variables.gene_number
layer1_dim = 8192
layer2_dim = 4096
output_dim = variables.PCA_components


class Encoder(nn.Module):
    def __init__(self, noiserate, noise_type, dropout_factor, device, input_order, pathway_proportion):
        super().__init__()

        self.device = device
        self.dropout_factor = dropout_factor
        self.noiserate = noiserate
        self.noise_type = noise_type
        pathways_genes = pd.read_csv(f'{variables.pathway_file}')
        self.pathways = pathways_genes.groupby('pathway')['gene'].apply(list).reset_index()
        self.pathway_noiseNum = int(self.pathways['pathway'].nunique() * pathway_proportion)

        # generate layers
        self.layer0 = self.generate_layer(input_dim, layer1_dim)
        self.layer1 = self.generate_layer(layer1_dim, layer2_dim)
        self.layer2 = self.generate_layer(layer2_dim, output_dim)
        self.layers = [self.layer0, self.layer1, self.layer2]

        # List of input genes in order
        self.input_order = input_order
        # For each gene, only keep everything up to the first dot
        self.input_order = [gene.split('.')[0] for gene in self.input_order]


    def forward(self, x):
        for i in range(0,3):
            x = self.add_noise(x, i)
            x = self.layers[i](x)
        return x

    def val_forward(self, x):
        for i in range(0,3):
            x = self.layers[i](x)
        return x


    def add_noise(self, x, i):

        if self.noise_type == 'masking':
            mask = torch.rand(x.size()) > self.noiserate
            mask = mask.to(self.device)
            return x * mask.float()
        elif self.noise_type == 'gaussian':
            noise = torch.randn(x.size()) * self.noiserate
            noise = noise.to(self.device)
            return x + noise
        elif self.noise_type == 'pathway':
            # Randomly select a number of pathways to add noise to equal to self.pathway_noiseNum
            pathways = self.pathways['pathway'].sample(n=self.pathway_noiseNum, replace=False)
            
            # If i == 0, add noise to the first layer
            for pathway in pathways:
                indices = [self.input_order.index(gene) for gene in self.pathways[self.pathways['pathway'] == pathway]['gene'].values[0]]
                indices = torch.tensor(indices, dtype=torch.long)

                if i == 0:
                    x = self.perturb(x, indices)
                else:
                    j = 1
                    while j <= i:
                        weights = self.layers[j-1][0].weight.T[indices]
                        indices = torch.argmax(weights, dim=1)
                        indices = indices.to_device(self.device)
                        j += 1
                    x = self.perturb(x, indices)
            return x



    def perturb(self, x, indices):
        x = x.T
        modes = ['over', 'under', 'mask']
        random_mode = modes[torch.randint(0, 3, (1,)).item()]

        if random_mode == 'over':
            x[indices] *= (1 + self.noiserate)
        elif random_mode == 'under':
            x[indices] *= (1 - self.noiserate)
        elif random_mode == 'mask':
            x[indices] = 0
        return x.T


    
    def generate_layer(self, first_dim, second_dim):
        layer = nn.Sequential(
            nn.Linear(first_dim, second_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_factor)
        )
        # Initialize weights
        nn.init.xavier_uniform_(layer[0].weight)
        nn.init.zeros_(layer[0].bias)
        return layer



class Decoder(nn.Module):
    def __init__(self, dropout_factor):
        super().__init__()

        # generate list of decoders
        self.decoders = nn.Sequential(
            nn.Linear(output_dim, layer2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_factor),
            nn.Linear(layer2_dim, layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_factor),
            nn.Linear(layer1_dim, input_dim))

    
    def forward(self, x):
        return self.decoders(x)



class DeepSDAE(nn.Module):

    def __init__(self, noiserate, noise_type, dropout_factor, device, input_order, pathway_proportion, split=False):
        super().__init__()

        self.noiserate = noiserate
        self.noise_type = noise_type
        self.dropout_factor = dropout_factor
        self.split = split

        self.encoder = Encoder(noiserate, noise_type, dropout_factor, device, input_order, pathway_proportion)
        self.decoder = Decoder(dropout_factor)
        
        # tie weights
        self.decoder.decoders[0].weight = nn.Parameter(self.encoder.layers[2][0].weight.t())
        self.decoder.decoders[3].weight = nn.Parameter(self.encoder.layers[1][0].weight.t())
        self.decoder.decoders[6].weight = nn.Parameter(self.encoder.layers[0][0].weight.t())

        if split:
            self.encoder.to('cuda:0')
            self.decoder.to('cuda:1')


    def forward(self, x):
        x = self.encoder(x)
        if self.split:
            x = x.to('cuda:1')
        x = self.decoder(x)
        return x


    def val_forward(self, x):
        x = self.encoder.val_forward(x)
        if self.split:
            x = x.to('cuda:1')
        x = self.decoder(x)
        return x


    def encode(self, x):
        return self.encoder.val_forward(x)