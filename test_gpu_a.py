from models.birefnet import BiRefNet

# Use codes and weights locally
import torch
from utils import check_state_dict

birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load('BiRefNet-general-epoch_244.pth', map_location='cpu')
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)