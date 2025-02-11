# This Python file uses the following encoding: utf-8

import numpy as np
import torch

if __name__ == "__main__":
    data = [[1, 2], [3, 4]]

    direct_tensor = torch.tensor(data)
    print("Direct tensor:", direct_tensor)

    numpy_data = np.array(data)
    numpy_tensor = torch.from_numpy(numpy_data)
    print("numpy data:", numpy_data)
    print("numpy tensor:", numpy_tensor)

    numpy_tensor.add_(5)
    print("numpy data:", numpy_data)
    print("numpy tensor:", numpy_tensor)
