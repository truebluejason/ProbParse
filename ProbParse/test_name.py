import sys
import torch

import const
from data_loader.json import load_json
from model.name import NameVAE

"""
Usage: python ProbParse/test/name.py <Config File Path>
"""
TEST_STRINGS = ['mike teng', 'frank donald wood', 'yung, dylan', 'lavington, jonathan wilder', 'jason j. yoo', 'li, gwen g.']

def predict(vae, data):
    with torch.no_grad():
        x = vae.permitted_set.to_one_hot_tensor(data)
        result = vae.predict(x)
    for full_name, parsed in zip(data, result):
        print(f"Full Name: {full_name}")
        print(f"- First/Middle/Last Name + Format: {parsed[0]}/{parsed[1]}/{parsed[2]} + {parsed[3]}")

if __name__ == "__main__":
    config = load_json(sys.argv[1])
    SESSION_NAME = config['session_name']
    vae = NameVAE(hidden_size=config['hidden_size'], num_layers=1, test_mode=True).to(const.DEVICE)
    vae.load_state_dict(torch.load(f"ProbParse/nn_model/{SESSION_NAME}", map_location=const.DEVICE))

    predict(vae, TEST_STRINGS)
