import string
import torch

MAX_NAME_LENGTH = 35
MAX_FIRST_NAME_LENGTH = 10
MAX_MIDDLE_NAME_LENGTH = 10
MAX_LAST_NAME_LENGTH = 10

EOS_TOKEN = "1"
PAD_TOKEN = "2"
LATENT_LETTERS = string.ascii_lowercase + EOS_TOKEN + PAD_TOKEN
PERMITTED_LETTERS = LATENT_LETTERS + " .,-'"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
