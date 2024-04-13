from models.ssdmobilenet_v2 import Model
from utils.hyper_params import get_hyper_params

# hyper parameters
hyper_params  = get_hyper_params(model='mobilenet_v2')

# model
model = Model(hyper_params)