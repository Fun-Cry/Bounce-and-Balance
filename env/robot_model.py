import numpy as np

design_rw = {
    "model": "design_rw",
    "urdfpath": "assets/hopper_rev08/urdf/hopper_rev08.urdf",
    "init_q": [-30 * np.pi / 180, -120 * np.pi / 180, -150 * np.pi / 180, 120 * np.pi / 180],
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "aname": ["q0", "q2", "rw1", "rw2", "rw3"],
    "a_kt": np.array([1.73, 1.73, 0.106, 0.106, 0.0868]),
    "inertia": np.array([[0.0754, 0.00016,  0.0022],
                         [0.00016, 0.0459, -0.00008],
                         [0.0022, -0.00008, 0.0771]]),
    "rh": -np.array([0.02663, 0.04436, 6.6108]) / 1000,
    "S": np.array([[1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]]),
    "hconst": 0.27,
    "n_a": 5,
    "ks": 3000,
    "springpolarity": 1,
    "k": 5000,
    "k_k": [45, 45 * 0.02],
    "mu": 2
}

model_dict = {
    "design_rw": design_rw
}
