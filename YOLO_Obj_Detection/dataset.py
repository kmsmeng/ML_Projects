import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

def VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transfrom = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])

        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.tranform:
            image, boxes = self.transform(image, boxes)
        
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # the x and y coordinates along with height and width label of images are normalized to scale of 1

            # row(i), column(j)
            
            '''
            # the way this is coded will only allow the x and y cooridntes of range [0, 1) so that is all value from 0 to 1 except 1. because at coordinte value of 1, we will get the row or column value as equal to (number of total cell i.e. `self.S`) but since the i and j are 0th index the i and j max value are (self.S - 1) so the coordintes will be neagtive when resizign for the instance that the coordintes are 1
            '''
            i, j = int(self.S * y), int(self.S * x)

            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (self.S * width, self.S * height)

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1
        
        return image, label_matrix