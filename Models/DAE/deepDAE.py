import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

import torch.nn as nn
import torch
import pandas as pd
import variables

input_dim = variables.gene_number
layer1_dim = 8192
layer2_dim = 4096
output_dim = variables.PCA_components
inverse = variables.inverse


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
        self.layers = nn.ModuleList([self.layer0, self.layer1, self.layer2])

        # List of input genes in order
        self.input_order = input_order


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
            
            perturbed_x = x.clone()

            # If i == 0, add noise to the first layer
            for pathway in pathways:
                indices = [self.input_order.index(gene) for gene in self.pathways[self.pathways['pathway'] == pathway]['gene'].values[0]]
                indices = torch.tensor(indices, dtype=torch.long)

                if i == 0:
                    perturbed_x = self.perturb(perturbed_x, indices)
                else:
                    j = 1
                    while j <= i:
                        weights = self.layers[j-1][0].weight.T[indices]
                        indices = torch.argmax(weights, dim=1)
                        indices = indices.to(self.device)
                        j += 1
                    perturbed_x = self.perturb(perturbed_x, indices)
            return perturbed_x
        elif self.noise_type == 'not_pathway':
            # Select all pathways
            pathways = self.pathways['pathway']
            
            perturbed_x = x.clone()

            # If i == 0, add noise to the first layer
            for pathway in pathways:
                indices = [self.input_order.index(gene) for gene in self.pathways[self.pathways['pathway'] == pathway]['gene'].values[0]]
                indices = torch.tensor(indices, dtype=torch.long)

                if i == 0:
                    perturbed_x = self.inverse_perturb(perturbed_x, indices)
                else:
                    j = 1
                    while j <= i:
                        weights = self.layers[j-1][0].weight.T[indices]
                        indices = torch.argmax(weights, dim=1)
                        indices = indices.to(self.device)
                        j += 1
                    perturbed_x = self.inverse_perturb(perturbed_x, indices)
            return perturbed_x



    def perturb(self, x, indices):
        x_copy = x.clone().T
        modes = ['over', 'under']
        random_mode = modes[torch.randint(0, 2, (1,)).item()]

        if not inverse:
            if random_mode == 'over':
                x_copy[indices] *= (1 + self.noiserate)
            elif random_mode == 'under':
                x_copy[indices] *= (1 - self.noiserate)
            return x_copy.T
        else:
            mask = torch.ones(x_copy.size(0), dtype=torch.bool, device=x_copy.device)
            mask[indices] = False
            if random_mode == 'over':
                # All values except the indices are multiplied by 1 + noiserate
                x_copy[mask] *= (1 + self.noiserate)
            elif random_mode == 'under':
                x_copy[mask] *= (1 - self.noiserate)
            return x_copy.T
        
    def inverse_perturb(self, x, indices):
        x_copy = x.clone().T

        mask = torch.ones(x_copy.size(0), dtype=torch.bool, device=x_copy.device)
        mask[indices] = False

        # Under-expressing all genes except those in the pathway
        x_copy[mask] *= (1 - self.noiserate)

        # Adding a small gaussian noise to all genes for natural variation
        noise = torch.randn(x_copy.size()) * (self.noiserate / 2)
        noise = noise.to(self.device)
        x_copy += noise

        # Mimicking a random batch effect
        modes = ['over', 'under']
        random_mode = modes[torch.randint(0, 2, (1,)).item()]
        if random_mode == 'over':
            x_copy += 0.1
        elif random_mode == 'under':
            x_copy -= 0.1

        return x_copy.T


    
    def generate_layer(self, first_dim, second_dim):
        layer = nn.Sequential(
            nn.Linear(first_dim, second_dim),
            nn.ELU(),
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
            nn.ELU(),
            nn.Dropout(dropout_factor),
            nn.Linear(layer2_dim, layer1_dim),
            nn.ELU(),
            nn.Dropout(dropout_factor),
            nn.Linear(layer1_dim, input_dim))

    
    def forward(self, x):
        return self.decoders(x)



class DeepDAE(nn.Module):

    def __init__(self, noiserate, noise_type, dropout_factor, device, input_order, pathway_proportion=0.1, split=False):
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
    



class TrainedEncoder(nn.Module):
    def __init__(self, dropout_factor):
        super().__init__()
        self.dropout_factor = dropout_factor

        # Define the layers exactly like the original encoder, but no extra logic
        self.layer0 = self.generate_layer(input_dim, layer1_dim)
        self.layer1 = self.generate_layer(layer1_dim, layer2_dim)
        self.layer2 = self.generate_layer(layer2_dim, output_dim)

        self.layers = nn.ModuleList([self.layer0, self.layer1, self.layer2])

    def generate_layer(self, first_dim, second_dim):
        layer = nn.Sequential(
            nn.Linear(first_dim, second_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_factor)
        )
        return layer
    
    def forward(self, x):
        for i in range(0,3):
            x = self.layers[i](x)
        return x