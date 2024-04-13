import json

backbone = "mobilenet_v2"

SSD =   {
    "mobilenet_v2": {
        "img_size": 1024,
        "feature_map_shapes": [128, 64, 32, 16, 8, 4],
        "aspect_ratios": [
            [1., 2., 1./2., 5., 1./5.],
            [1., 2., 1./2., 5., 1./5.],
            [1., 2., 1./2., 5., 1./5.],
            [1., 2., 1./2., 5., 1./5.],
            [1., 2., 1./2., 5., 1./5.],
            [1., 2., 1./2., 5., 1./5.]
        ],
        "use_custom_scale": True,
        "scale_min": 0.2,
        "scale_max": 0.9,
        "scale": [0.05, 0.1, 0.2, 0.4, 0.7, 1, 1.5],
        "trainable": True,
        "num_trainable": None
    }
}

def get_hyper_params(model='mobilenet_v2', **kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs = any value could be updated in the hyper_params

    outputs:
        hyper_params = dictionary
    """
    hyper_params = SSD[model]
    hyper_params["detection"] = "None" # "None" / "FPN" / "BiFPN" / "PAFPN" / "NASFPN"
    hyper_params["feature_fusion"] = None #
    hyper_params["dataset"] = 0 # dut, tilda, daffodil, thesis, combined
    hyper_params["iou_threshold"] = 0.5
    hyper_params["neg_pos_ratio"] = 3 # neg:pos 3:1 ratio
    hyper_params["loc_loss_alpha"] = 1 # weight for the localization loss
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    hyper_params["use_focal"] = False
    hyper_params["alpha"] = 2.0
    hyper_params["gamma"] = 0.25
    hyper_params["batch_size"] = 8
    hyper_params["epochs"] = 100
    hyper_params["lr"] = 1e-05
    hyper_params["patience"] = 20
    # overwrite any parameters
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value

    return hyper_params

if __name__ == '__main__':
    hyper_params = get_hyper_params(model='mobilenet_v2')
    print(json.dumps(hyper_params, indent=4))