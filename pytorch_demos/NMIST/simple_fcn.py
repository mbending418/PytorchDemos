import os
import json
import torch

from typing import Tuple


class FCNetImageClassifier(torch.nn.Module):
    """
    simple FCN Classifier
    takes in an image of size "image_size"
    has a single hidden layer of size "hidden_layer_size"
    the hidden layer is followed by a RELU
    regresses to an output layer size "output_categories"
    """
    image_size: Tuple[int, int]
    hidden_layer_size: int
    output_categories: int

    def __init__(
        self,
        image_size: Tuple[int, int] = (28, 28),
        hidden_layer_size: int = 15,
        output_categories: int = 10,
    ):
        super(FCNetImageClassifier, self).__init__()

        self.image_size = image_size
        self.hidden_layer_size = hidden_layer_size
        self.output_categories = output_categories

        self.fc1 = torch.nn.Linear(
            in_features=image_size[0] * image_size[1], out_features=hidden_layer_size
        )
        self.fc2 = torch.nn.Linear(
            in_features=hidden_layer_size, out_features=output_categories
        )

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

    def save_model(self, file_name: str, folder: str = "."):
        """
        save pickeled model and config file to output folder

        :param file_name: what to call the output model
        :param folder: the folder to save it to
        :return:
        """
        model_file_name = os.path.join(folder, file_name + ".pt")
        json_file_name = os.path.join(folder, file_name + "_configs.json")
        config_dict = {
            "image_size": self.image_size,
            "hidden_layer_size": self.hidden_layer_size,
            "output_categories": self.output_categories,
        }
        with open(json_file_name, "w") as jfn:
            json.dump(config_dict, jfn, indent=4)

        torch.save(self.state_dict(), model_file_name)

    @staticmethod
    def load_model(file_name: str, folder: str = ".") -> "FCNetImageClassifier":
        """
        load a picked model

        :param file_name: the name of the model
        :param folder: the folder to look in
        :return: the loaded model
        """
        model_file_name = os.path.join(folder, file_name + ".pt")
        json_file_name = os.path.join(folder, file_name + "_configs.json")
        with open(json_file_name, "r") as jfn:
            config = json.load(jfn)

        net = FCNetImageClassifier(**config)
        net.load_state_dict(torch.load(model_file_name))
        return net
