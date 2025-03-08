import torch

def eval_model(model, dataloader, device):
    '''Passes the Data into the model and return the encoded data from the autoencoder and the label data'''
    model.eval()

    encoded_data = []
    label_data = []
    with torch.inference_mode():
        for image, label in dataloader:
            image = image.to(device)

            output = model(image)
            encoded_output = output[0]
            encoded_data.append(encoded_output)
            label_data.append(label)

    encoded_data = torch.concat(encoded_data)
    label_data = torch.concat(label_data)

    return encoded_data, label_data