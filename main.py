
import cv2
import torch
from torch.utils.data import DataLoader

from dataset import FightDataset
from transform import Compose, ToTensor, Resize
from model import MyNet
torch.cuda.set_device(0) 
transform_ = Compose([
    Resize((112,112)),
    ToTensor()
    
])
xx = FightDataset("./fight_classify", tranform=transform_)

dataloader = DataLoader(xx, batch_size=1, shuffle=True)
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch)
#     print(sample_batched["image"].size())
dev = torch.device("cuda:0")
model = MyNet().to(dev)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

for t in range(20):
    # Forward pass: Compute predicted y by passing x to the model
    for i_batch, sample_batched in enumerate(dataloader):
        image = sample_batched["image"]
        label = sample_batched["label"]
        # label = torch.transpose(label, 0,1)
        y_pred = model(image)
        # print(y_pred)
        # print(label)
        # Compute and print loss
        loss = criterion(y_pred, label)
        if i_batch % 2000 == 1999:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Result: {}".format(model.string()))
