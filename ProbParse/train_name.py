import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import const
import test_name
from data_loader.json import save_json
from data_loader.name import NameDataset
from model.name import NameVAE


# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='UNNAMED_SESSION', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.0005, type=float)
parser.add_argument('--batch_size', help='Size of the batch training on', nargs='?', default=256, type=int)
parser.add_argument('--num_epochs', help='Number of epochs', nargs='?', default=3000, type=int)
parser.add_argument('--continue_training', help='Boolean argument whether to continue training an existing model',
                     nargs='?', default=False, type=bool)
args = parser.parse_args()
SESSION_NAME = args.name
to_save = {
    'session_name': SESSION_NAME,
    'hidden_size': args.hidden_size,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
    'learning_rate': args.lr,
}
save_json(f'config/{SESSION_NAME}.json', to_save)


def loss_function(x, x_hat, log_prob, kls, kl_weights):
    binary_cross_entropy = F.binary_cross_entropy(x_hat, x, reduction='mean')
    reinforce = log_prob * binary_cross_entropy.detach()
    kl_sum = kls['format']*kl_weights['format'] + kls['fname']*kl_weights['fname'] + kls['mname']*kl_weights['mname'] + kls['lname']*kl_weights['lname']
    loss = binary_cross_entropy + reinforce - reinforce.detach() + kl_sum
    return loss

def train_one_epoch(vae, data, kl_weights, print_every=3):
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    data = iter(data)
    epoch_loss = 0.
    for i in range(len(data)):
        names = next(data)
        x = vae.permitted_set.to_one_hot_tensor(names)
        x_hat, log_prob, kls = vae.forward(x)
        loss = loss_function(x, x_hat, log_prob, kls, kl_weights)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        if (i-1) % print_every == 0: print(f"Batch {i} Loss: {loss.item()}")
    return epoch_loss/len(data)

def test(vae, data, kl_weights):
    data = iter(data)
    epoch_loss = 0.
    with torch.no_grad():
        for _ in range(len(data)):
            names = next(data)
            x = vae.permitted_set.to_one_hot_tensor(names)
            x_hat, log_prob, kls = vae.forward(x)
            loss = loss_function(x, x_hat, log_prob, kls, kl_weights)
            epoch_loss += loss.item()
    return epoch_loss/len(data)

def strict_accuracy(vae, data):
    data = iter(data)
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for _ in range(len(data)):
            names = next(data)
            x = vae.permitted_set.to_one_hot_tensor(names)
            x_hat, _, _ = vae.forward(x)
            original = vae.permitted_set.to_words(x)
            reconstructed = vae.permitted_set.to_words(x_hat)
            num_total += len(original)
            for o, r in zip(original, reconstructed):
                if o == r: num_correct += 1
    return num_correct / num_total

def train_test_split(csv_path, random_state=1):
    df = pd.read_csv(csv_path)
    df_shuffle = df.sample(n=len(df), random_state=random_state)
    train_df, test_df = df_shuffle[:int(len(df) * 0.9)], df_shuffle[int(len(df) * 0.9):]
    train_dataset = NameDataset(train_df, 'name', max_name_length=const.MAX_NAME_LENGTH)
    test_dataset = NameDataset(test_df, 'name', max_name_length=const.MAX_NAME_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def plot_losses(train_losses, test_losses, folder: str, filename: str):
    x = list(range(len(train_losses)))
    plt.plot(x, train_losses, 'b--', label="Training Loss")
    plt.plot(x, test_losses, 'ro', label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()

if __name__ == "__main__":
    train_data, test_data = train_test_split('data/balanced.csv')
    vae = NameVAE(hidden_size=args.hidden_size, num_layers=1, test_mode=False).to(const.DEVICE)
    if args.continue_training:
        vae.load_state_dict(torch.load(f"ProbParse/nn_model/{SESSION_NAME}", map_location=const.DEVICE))
    kl_weights = {'fname': 0., 'mname': 0., 'lname': 0., 'format': 0.}

    train_losses, test_losses = [], []
    for i in range(args.num_epochs):
        print(f"Epoch {i}")
        train_loss = train_one_epoch(vae, train_data, kl_weights)
        test_loss = test(vae, test_data, kl_weights)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Strict Validation Reconstruction Accuracy: {strict_accuracy(vae, test_data)}")
        print(f"Parsing Sample Names")
        test_name.predict(vae, test_name.TEST_STRINGS)
        plot_losses(train_losses, test_losses, folder="ProbParse/result", filename=f"{SESSION_NAME}.png")
        torch.save(vae.state_dict(), f"ProbParse/nn_model/{SESSION_NAME}")
