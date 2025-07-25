import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# x = torch.ones(3,3,requires_grad=True)
x = torch.rand(3,3,requires_grad=True)
# print(x)

# y = x+5
# # print(y)

# z = y*y
# out = z.mean()
# out.backward()
# # print('z: ', z)
# # print('out: ', out)

# # b_z = z.backward()
# # print("b_z: ", b_z)

# print(x)
# print(x.grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())


mnist_transfor = transforms.Compose([transforms.ToTensor()])
