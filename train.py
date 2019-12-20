import os
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import CharactersDataset
from model import CharacterClassifier
from torch.utils.tensorboard import SummaryWriter
from utils import *

# save Tensorboard logs somewhere
log_dir = "logs/lr_{}_batchsize_{}".format(args.learning_rate, args.batch_size)
writer = SummaryWriter(log_dir)

# we need to save the models somewhere
model_fn = "epoch_{}_lr_{}_batchsize_{}"
if not os.path.exists("models"):
    os.mkdir("models")

# are we using GPU or CPU?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Cuda available? {}".format(torch.cuda.is_available()))
print("Device: {}".format(device))

# initialize model, dataset, optimizers, and data loaders
model = CharacterClassifier(num_classes = 36, pretrained = False).to(device)

train_dataset = CharactersDataset(data_root = os.path.join("data", "characters", "train"))
val_dataset = CharactersDataset(data_root = os.path.join("data", "characters", "val"))
test_dataset = CharactersDataset(data_root = os.path.join("data", "characters", "test"))

print("\nTrain dataset size: {}".format(len(train_dataset)))
print("Validation dataset size: {}".format(len(val_dataset)))
print("Test dataset size: {}".format(len(test_dataset)))

optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay, betas = (0.9, 0.999))
loss_func = nn.BCELoss(reduction = 'sum').to(device)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 1)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = 1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 1)

# train and validate the model
print("\nTraining with settings: ")
print("\tLearning rate: ", args.learning_rate)
print("\tWeight decay: ", args.weight_decay)
print("\tBatch size: ", args.batch_size)
print("\tEpochs: ", args.num_epochs)

lowest_validation_loss = None
lowest_validation_epoch = None

for epoch in range(args.num_epochs):
    # train the model
    for batch_id, samples in enumerate(train_loader):
        labels = samples['labels'].to(device)

        imgs = samples['imgs'].to(device)
        pred = model(imgs).to(device)
        val = torch.max(pred)
        #print(pred, labels)
        print(val.item())

        loss = loss_func(pred, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + batch_id)
        if batch_id % 1 == 0:
            print("(train) => Epoch {}/{} - Batch {}/{} - Loss: {}".format(epoch, args.num_epochs, batch_id, len(train_loader), loss.item()))

    # validate the model
    validation_loss = 0.0
    for batch_id, samples in enumerate(val_loader):
        labels = samples['labels'].to(device)

        imgs = samples['imgs'].to(device)
        pred = model(imgs).to(device)

        loss = loss_func(pred, labels)
        validation_loss += loss.item()

        if batch_id % 1 == 0:
            print("(validation) => Epoch {}/{} - Batch {}/{} - Loss: {}, Running loss: {}".format(epoch, args.num_epochs, batch_id, len(val_loader), loss.item(), validation_loss))
    validation_loss /= len(val_loader)
    print("(validation) => Epoch {}/{} - Batch {}/{} - Avg. Loss: {}".format(epoch, args.num_epochs, batch_id, len(val_loader), validation_loss))
    writer.add_scalar('Validation loss', validation_loss, epoch * len(val_loader) + batch_id)

    # we found a new 'best' model in terms of validation loss, so save it to disk
    lowest_validation_loss_string = "None" if lowest_validation_loss is None else round(lowest_validation_loss, 5)
    if lowest_validation_loss is None or validation_loss < lowest_validation_loss:
        # remove the old 'best' model
        if lowest_validation_epoch is not None:
            old_model_path = os.path.join("models", model_fn.format(lowest_validation_epoch, args.learning_rate, args.batch_size))
            if os.path.exists(old_model_path):
                os.remove(old_model_path)

        # save new best model
        print("** New best model ; Epoch {} ; Old best val loss: {}, New best val loss: {}".format(epoch, lowest_validation_loss_string, round(validation_loss, 5)))
        torch.save(model.state_dict(), os.path.join("models", model_fn.format(epoch, args.learning_rate, args.batch_size)))
        lowest_validation_loss = validation_loss
        lowest_validation_epoch = epoch

# test the model
test_loss = 0.0
for batch_id, samples in enumerate(test_loader):
    labels = samples['labels'].to(device)

    imgs = samples['imgs'].to(device)
    pred = model(imgs).to(device)

    loss = loss_func(pred, labels)
    test_loss += loss.item()

    if batch_id % 1 == 0:
        print("(test) => Batch {}/{} - Loss: {}, Running loss: {}".format(batch_id, len(test_loader), loss.item(), test_loss))
test_loss /= len(test_loader)
print("(test) => Batch {}/{} - Avg. Loss: {}".format(batch_id, len(test_loader), test_loss))
