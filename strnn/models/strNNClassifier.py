import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from strnn.models.strNN import StrNN
from numpy.random import binomial
from strnn.models.strNN import MaskedLinear
from strnn.models.model_utils import NONLINEARITIES

SUPPORTED_DATA_TYPES = ['binary', 'gaussian']


class StrNNClassifier(strNN):
    def __init__(self, num_features, num_classes, hidden_sizes, adjacency_matrix, init_type, activation='relu'):
        # Initialize the StrNN with nin = nout to maintain the autoregressive property
        super().__init__(
            nin=num_features,
            hidden_sizes=hidden_sizes,
            nout=num_features,  # Using num_features as nout to maintain autoregressive property
            opt_type='greedy',
            opt_args={'var_penalty_weight': 0.0},
            precomputed_masks=None,
            adjacency=adjacency_matrix,
            activation=activation,
            init=init_type
        )

        # Additional layer to transform the autoregressive output to class probabilities
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the structured part of the network
        transformed_features = super(StrNNClassifier, self).forward(x)
        # The linear classifier layer maps the structured features to class logits
        class_logits = self.classifier(transformed_features)
        return class_logits
