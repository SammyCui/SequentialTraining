import torch
import torchvision
import pickle

with open('/Users/xuanmingcui/Downloads/outputs_target.pickle', 'rb') as file:
    data = pickle.load(file)

print(data['outputs'])
print(data['targets'])

criterion = torch.nn.CrossEntropyLoss()

outputs = data['outputs']
targets = data['targets']

print(criterion(torch.tensor(outputs), torch.tensor(targets)))
