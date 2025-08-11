# Modified from: https://github.com/isl-org/VI-Depth/blob/main/modules/midas/base_model.py
# Original work licensed under the MIT License
# Copyright (c) 2023 Intelligent Systems Lab Org
# Modifications Copyright (c) 2025 Hongjie Zhang

import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        if "state_dict" in parameters:
            state_dict = parameters["state_dict"]
            new_state_dict = {}
            for key in state_dict.keys():
                if key[0:6] == "model.":
                    new_state_dict[key[6:]] = state_dict[key]

            self.load_state_dict(new_state_dict)

        else:
            self.load_state_dict(parameters)

    def save(self, path):
        """Save model to file.

        Args:
            path (str): file path
        """
        torch.save(self.state_dict(), path)
