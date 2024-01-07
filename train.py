import json
import torch
import net
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ObjLocDataset(Dataset):
    def __init__(self,file_path):
        file_ptr = open(file_path,'r')
        file_content = file_ptr.read()
        file_json = json.loads(file_content)
        file_ptr.close()

        self.inputs = []
        self.outputs = []


        for sample in file_json:
            image = Image.open(sample['image_path'])
            coords = torch.tensor([
                sample['x'],
                sample['y'],
                sample['width'],
                sample['height']]
            ).to(torch.float)
            image_tensor = torch.from_numpy(np.array(image)).to(torch.float)
            self.inputs.append(image_tensor)
            self.outputs.append(coords)
        
    def get_sample_shape(self):
        return self.inputs[0].shape
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,index):
        return (self.inputs[index], self.outputs[index])


def plot_bounding_box(image, bbox):

    image = image.detach().numpy()
    print(type(image))
    image = Image.fromarray(image[0])

    fig, ax = plt.subplots()
    
    ax.imshow(image)


    x, y, width, height = bbox
    rect = patches.Rectangle((x,y), width, height, linewidth=2, edgecolor="r",facecolor='none')
    
    ax.add_patch(rect)
    plt.show()

def train(file_path: str):

    dataset = ObjLocDataset(file_path)
    dataLoader = DataLoader(dataset, batch_size=1)

    sample_shape = dataset.get_sample_shape()
    
    model = net.Net(sample_shape[0]*sample_shape[1], 4)
    optim = torch.optim.Adam(model.parameters(),lr=5e-4)

    criterion = torch.nn.MSELoss()

    EPOCHS = 100

    # training loop

    print("Training on Samples...")

    for epoch in range(EPOCHS):
        epoch_loss = []
        for idx,data in enumerate(dataLoader):

            optim.zero_grad()
            
            data_input = data[0]
            data_label = data[1]

            output = model(data_input)
            loss = criterion(data_label, output)

            loss.backward()
            optim.step()

            epoch_loss.append(loss)


        epoch_loss = np.array(epoch_loss)
        print(f"Epoch {epoch}: ",epoch_loss)


    # eval

    model.eval()

    sample = next(iter(dataLoader))
    net_op = model(sample[0])
    net_op_np = net_op.detach().numpy()
    plot_bounding_box(sample[0], net_op[0])

if __name__ == "__main__":
    train("./samples/dataset.json")
