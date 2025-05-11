import torch

def eval_model(model, dataloader, device):
    '''Returns all the encoded data from the model and their true respective labels'''

    with torch.inference_mode():
        model.eval()

        encoded_list = []
        label_list = []

        # Iterate through all the images and labels of the dataloader
        for image, label in dataloader:

            # Put image into GPU for inference, forward pass into the model
            image= image.to(device)

            # forward pass to the model to get the encoded data
            encoded = model(image)[0]
            
            # Append the encoded data and their respective true labels into the list
            encoded_list.append(encoded)
            label_list.append(label)
            
        label_list = torch.concat(label_list)
        encoded_list = torch.concat(encoded_list)
        return encoded_list, label_list

