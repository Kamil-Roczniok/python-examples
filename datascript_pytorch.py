import torch
import torch.nn as nn
import torch.optim as optim

#Simple model
model = nn.Sequential(nn.Linear(10,5), nn.ReLU(), nn.Linear(5,1))

#Sample data
input_data = torch.randn(1,10)
output_data = model(input_data)

print(output_data)