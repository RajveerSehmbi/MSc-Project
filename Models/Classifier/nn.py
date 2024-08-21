import sys
sys.path.append('/vol/bitbucket/rs218/MSc-Project')

import torch.nn as nn
import torch
import variables

class Classifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_factor):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_factor),
            nn.Linear(hidden_dim, 34),
        )

        self.output = nn.Softmax(dim=1)

        # Initialize weights
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)
        nn.init.xavier_uniform_(self.layers[3].weight)
        nn.init.zeros_(self.layers[3].bias)

    def forward(self, x):
        x = self.layers(x)
        return x


class DAEClassifier(nn.Module):

    def __init__(self, dropout_factor):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(variables.gene_number, 8192),
            nn.ReLU(),
            nn.Dropout(dropout_factor),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_factor),
            nn.Linear(4096, variables.PCA_components),
            nn.ReLU(),
            nn.Dropout(dropout_factor),
            nn.Linear(variables.PCA_components, 1753),
            nn.ReLU(),
            nn.Dropout(dropout_factor),
            nn.Linear(1753, 34)
        )

        self.output = nn.Softmax(dim=1)

        # Load the weights from the pre-trained DAE
        pretrainedDAE = torch.load(f'{variables.DAE_model_path}/{variables.DAE_type}_deepDAE_encoder.pt')

        self.layers[0].weight = pretrainedDAE.layers[0][0].weight
        self.layers[0].bias = pretrainedDAE.layers[0][0].bias
        self.layers[3].weight = pretrainedDAE.layers[1][0].weight
        self.layers[3].bias = pretrainedDAE.layers[1][0].bias
        self.layers[6].weight = pretrainedDAE.layers[2][0].weight
        self.layers[6].bias = pretrainedDAE.layers[2][0].bias


        # Initialize random weights for the classifier layers
        nn.init.xavier_uniform_(self.layers[9].weight)
        nn.init.zeros_(self.layers[9].bias)
        nn.init.xavier_uniform_(self.layers[12].weight)
        nn.init.zeros_(self.layers[12].bias)

    def forward(self, x):
        x = self.layers(x)
        return x