import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from model import YOLOV1
from dataset import VOCDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from torch.utils.data import DataLoader

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# HYPERPARAMETERS
LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMEORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = 'overfit.pth.tar'
IMG_DIR = 'data/archive/images'
LABEL_DIR = 'data/archive/labels'

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        
        return img, bboxes
        
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    # model.train()

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss = loss.item())

    print(f'Mean Loss was {sum(mean_loss)/len(mean_loss)}')


def main():
    model = YOLOV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        'data/archive/8examples.csv',
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform
    )

    test_dataset = VOCDataset(
        'data/archive/test.csv',
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform
    )

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=PIN_MEMEORY,
                                  shuffle=True,
                                  drop_last=False) # drops the last batch if the number of data samples in the last batch is smaller then the batch size

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=PIN_MEMEORY,
                                 shuffle=True,
                                 drop_last=True)
    
    for epochs in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(train_dataloader, model, iou_threshold=0.5, threshold=0.4)

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint')

        print(f'Train mAP: {mean_avg_prec}')

        train_fn(train_dataloader, model, optimizer, loss_fn)



if __name__ == '__main__':
    main()
