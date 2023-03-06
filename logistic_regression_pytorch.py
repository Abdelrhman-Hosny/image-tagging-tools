'''
Class definition for Torch Logistic Regression model
'''

import torch

class LogisticRegressionPytorch(torch.nn.Module):
     def __init__(self, output_dim):
         super(LogisticRegressionPytorch, self).__init__()
         self.linear = torch.nn.LazyLinear(output_dim)
     def forward(self, x):
         return torch.sigmoid(self.linear(x))