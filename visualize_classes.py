import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from model import CharacterClassifier
from datasets import CharactersDataset

def load_characters(path):
    dataset = CharactersDataset(data_root = path, validate = True)
    num_points = len(dataset) # 500
    loader = torch.utils.data.DataLoader(dataset, batch_size = num_points, shuffle = False, num_workers = 1)
    batch = next(iter(loader))

    imgs = batch['imgs'].numpy()
    labels = batch['labels'].numpy()
    return imgs, labels

def plot_points(data, labels, chart_title):
    x = data[:, 0] # i.e. shape is (N, 2) because it's reduced to 2D, so the x points need to be the first column
    y = data[:, 1] # y points are the second column
    sactter = plt.scatter(x = x, y = y, c = labels, s = 9)
    plt.title("{}".format(chart_title))
    plt.show()

# are we using GPU or CPU?
# also, initialize a model and load trained weights
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Cuda available? {}".format(torch.cuda.is_available()))
print("Device: {}".format(device))
model = CharacterClassifier(num_classes = 36, pretrained = False).to(device)
model.load_state_dict(torch.load(os.path.join("models", "epoch_7_lr_5e-05_batchsize_512")))
model.to(device)

# load data
data_path = os.path.join("data", "characters", "test")
digits_data, digits_labels = load_characters(data_path)
digits_data_pretrained = model.model[0:8](torch.Tensor(digits_data).to(device)).cpu().view(digits_data.shape[0], -1).detach().numpy() # use layers 0:8 from the model for feature extraction, pass images through the model, come back, reshape to (N, feats), and convert from tensor to numpy array
digits_data = digits_data.reshape((digits_data.shape[0], 5776)) # reshape vanilla dataset from (N, 1, 76, 76) to (N, 5776) so we can pass them into TSNE

# do PCA dimension reduction first
print("Performing PCA dimensionality reduction on pretrained features...")
pca = PCA(n_components = 100)
digits_data_pretrained = pca.fit_transform(digits_data_pretrained)
print("PCA explained variance (top 10 components): {}...".format(pca.explained_variance_ratio_[0:10]))

# reduce dimensionality with TSNE
print("Processing TNSE for vanilla dataset (without using pretrained model)...")
digits_tsne_vanilla = TSNE(n_components = 2).fit_transform(digits_data) # reduce the original data to 2D
print("Processing TNSE for pretrained dataset (features extracted from pretrained model)...")
digits_tsne_pretrained = TSNE(n_components = 2).fit_transform(digits_data_pretrained) # reduce the output features from a model to 2D

# visualize the two plots
plot_points(digits_tsne_vanilla, digits_labels, "Vanilla Digits TSNE") # plot the TSNE points from the original dataset
plot_points(digits_tsne_pretrained, digits_labels, "Pretrained-model Digits TSNE") # plot the TSNE points from the pretrained model
