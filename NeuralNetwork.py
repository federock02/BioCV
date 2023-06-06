#Package import 
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import csv
from tqdm import tqdm
from skimage import io, color, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn
from sklearn.metrics import f1_score

# helper function for visualizing some samples of the images/mask to controll overlap
def image_viewer(output_classes, image, mask):
    
    x=mask[0].numpy()
    x*=output_classes # the function need the classes to be integer 
    io.imshow(color.label2rgb(x, image[0].numpy(), bg_label=0)) # set bkg transparent and shows only
    # other classes on top of the input png     
plt.show()

# dataset class loading input images and from two separate folders in numeric order   
class kidneyDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, output_classes, transform = None):
        self.image_dir = image_dir # getting image folder
        self.mask_dir = mask_dir  # getting segmentation forlder
        self.transform = transform # If there are transformtion to apply
        self.output_classes = output_classes

    def __len__(self):
        return int(len(glob.glob(os.path.join(self.image_dir,"*.npy")))/100) # number of images found in the folder

    def __getitem__(self,idx):
        img_name = os.path.join(self.image_dir,"%d.npy"%idx)
        img = np.load(img_name)
        tmp_img=np.ndarray((self.output_classes, img.shape[0],img.shape[1]),dtype=np.uint8)
        for i in range(self.output_classes):
            tmp_img[i,:,:]=img
        mask_name = os.path.join(self.mask_dir,"%d.npy"%idx)
        mask = np.load(mask_name)
        tmp_mask=np.ndarray((self.output_classes, mask.shape[0],mask.shape[1]),dtype=np.uint8)
        for i in range(self.output_classes):
            tmp_mask[i,:,:]=mask
        sample = {"image": tmp_img, "mask": tmp_mask} # matched image and mask

        if self.transform:
            sample=self.transform(sample) # eventual transformation to be made on the input data
        return sample
      
# custom function to transform input data to tensor of shape Nc, H, W
class ToTensor(object):
    
    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        mask_tensor = torch.from_numpy(mask).type(torch.FloatTensor)
        return {"image": img_tensor, "mask": mask_tensor}

# input normalization to have image and mask in [0 1] based on data range and classes
class Normalize(object):
 
    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        return {"image": img/255,
                "mask": mask/255}
    
# freezing model to not retrain everything given the low dimensionality of the dataset
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# load of pretrained model and change of last layer to match the number of classes to be predicted
# newly added layer will be trained
def createModel(output_classes):
    my_model = models.segmentation.fcn_resnet101(pretrained=True)
    set_parameter_requires_grad(my_model, feature_extracting= True)
    my_model.classifier[4] = nn.Conv2d(512, output_classes, kernel_size=(1, 1), stride=(1, 1))
    my_model.aux_classifier[4] = nn.Conv2d(256, output_classes, kernel_size=(1, 1), stride=(1, 1))
    my_model.train() 
    return my_model

# function for model training 
def train_model(model, criterion, dataloader, dataloader1, optimizer, metrics, bpath, num_epochs):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # logger
    fieldnames = ["epoch", "Train_loss", "Val_loss"] + \
        [f"Train_{m}" for m in metrics.keys()] + \
        [f"Val_{m}" for m in metrics.keys()]
    with open(os.path.join(bpath, "log.csv"), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
 
    for epoch in range(1, num_epochs+1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)
    
        batchsummary = {a: [0] for a in fieldnames}
 
        for phase in ["Train", "Val"]:
            if phase == "Train":
                model.train()  # Set model to training mode
                dataloaders=dataloader # Select dataset for training
            else:
                model.eval()   # Set model to evaluate mode
                dataloaders=dataloader1 #Select dataste for validation
 
            # Iterate over data.
            for sample in tqdm(iter(dataloaders)):
                inputs = sample["image"].to(device)
                masks = sample["mask"].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
 
                # track history if only in train
                with torch.set_grad_enabled(phase == "Train"):
                    outputs = model(inputs)
                    loss = criterion(outputs["out"], masks)
                    y_pred = outputs["out"].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == "f1_score":
                            batchsummary[f"{phase}_{name}"].append(
                                metric(y_true > 0, y_pred > 0, average="weighted"))
                        
                    # backward + optimize only if in training phase
                    if phase == "Train":
                        loss.backward()
                        optimizer.step()
            batchsummary["epoch"] = epoch
            epoch_loss = loss
            batchsummary[f"{phase}_loss"] = epoch_loss.item()
            print("{} Loss: {:.4f}".format(
                phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, "log.csv"), "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == "Val" and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
 
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# helper function for model semantic segmentation output
def decode_segmap(image, nc=3):
    label_colors = np.array([(0, 0, 0),
               (0, 255, 0), (0, 0, 255)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def main():
    torch.cuda.empty_cache()
    output_classes = 3
    #dataset and loader
    transformed_dataset_train = kidneyDataset(image_dir = "D:\Programming\BioCV\Project\PreprocessedImages\Images2D\Train\\", mask_dir = "D:\Programming\BioCV\Project\PreprocessedImages\Segmentations2D\Train\\", output_classes = output_classes, transform = transforms.Compose([Normalize(), ToTensor()]))
    transformed_dataset_val = kidneyDataset(image_dir = "D:\Programming\BioCV\Project\PreprocessedImages\Images\Validation2D\\", mask_dir = "D:\Programming\BioCV\Project\PreprocessedImages\Segmentations\Validation2D\\", output_classes = output_classes, transform = transforms.Compose([Normalize(), ToTensor()]))
    
    l_samples = len(transformed_dataset_train)
    for i in range(l_samples):
        sample = transformed_dataset_train[i]
        #print(i, sample["image"].shape, sample["mask"].shape)
    print(str(l_samples)+" samples loaded")
    
    dataloader = DataLoader(transformed_dataset_train, batch_size = 10, shuffle = True)
    dataloader1 = DataLoader(transformed_dataset_val, batch_size= 10, shuffle=False)
        
    # Model creation and criterion, optimizer and metric
    my_model = createModel(output_classes)
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.1)
    metrics = {"f1_score": f1_score}
    bpath = "D:\Programming\BioCV\Project\PreprocessedImages\\"
    # Model training 
    my_model_trained=train_model(my_model, criterion, dataloader, dataloader1, optimizer, metrics, bpath, num_epochs=4)
    
    torch.save(my_model_trained, "saved_model.pth")

    # Getting first batch of the training data to run the model and see its performance
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched["image"].size(),
          sample_batched["mask"].size())  
        break
        
    
    # Visualization of the model output on one example image from training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    my_inputs = sample_batched["image"].to(device)
    out = my_model_trained(my_inputs)["out"]
    out_c = out.cpu()
    om = torch.argmax(out_c[0],dim=0).numpy()
    rgb = decode_segmap(om)
    plt.imshow(rgb); plt.show()

main()
