import os
import json
import torch

from typing import Tuple


class ConvNet(torch.nn.Module):
    """
    simple CNN Classifier
    the model:
        takes in an image of size "image_size"
        two convolutional layers each followed by RELU
        a max pool layer
        a hidden layer
        and regresses to an output layer size "output_categories"
    """

    _features_layer_1 = 32
    _kernel_size_layer_1 = 3
    _stride_layer_1 = 1

    _features_layer_2 = 64
    _kernel_size_layer_2 = 3
    _stride_layer_2 = 1

    _max_pool_kernel_size = 2

    _hidden_fc_layer_size = 128

    def __init__(self, image_size: Tuple[int, int], output_categories: int):
        super(ConvNet, self).__init__()

        self.image_size = image_size
        self.output_categories = output_categories

        # how big each image feature is
        feature_size = list(image_size)

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self._features_layer_1,
            kernel_size=self._kernel_size_layer_1,
            stride=self._stride_layer_1,
        )
        # after a convolutional layer the size of each feature shrinks by the kernel_size*stride + 1
        feature_size = [
            feature_dim - self._kernel_size_layer_1 * self._stride_layer_1 + 1
            for feature_dim in feature_size
        ]

        self.conv2 = torch.nn.Conv2d(
            in_channels=self._features_layer_1,
            out_channels=self._features_layer_2,
            kernel_size=self._kernel_size_layer_2,
            stride=self._stride_layer_2,
        )
        # after a convolutional layer the size of each feature shrinks by the kernel_size*stride + 1
        feature_size = [
            feature_dim - self._kernel_size_layer_2 * self._stride_layer_2 + 1
            for feature_dim in feature_size
        ]

        self.max_pool = torch.nn.functional.max_pool2d
        feature_size = [
            (feature_dim // self._max_pool_kernel_size) for feature_dim in feature_size
        ]

        self.dropout1 = torch.nn.Dropout(0.25)

        self.flatten = torch.flatten
        # flatten takes all the features from the most recent layer and flattens them into a 1D tensor
        features = feature_size[0] * feature_size[1] * self._features_layer_2

        self.fc1 = torch.nn.Linear(
            in_features=features, out_features=self._hidden_fc_layer_size
        )
        self.dropout2 = torch.nn.Dropout(0.5)

        self.fc2 = torch.nn.Linear(
            in_features=self._hidden_fc_layer_size, out_features=output_categories
        )

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x, self._max_pool_kernel_size)
        x = self.dropout1(x)
        x = self.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
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
            "output_categories": self.output_categories,
        }
        with open(json_file_name, "w") as jfn:
            json.dump(config_dict, jfn, indent=4)

        torch.save(self.state_dict(), model_file_name)

    @staticmethod
    def load_model(file_name: str, folder: str = ".") -> "ConvNet":
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

        net = ConvNet(**config)
        net.load_state_dict(torch.load(model_file_name))
        return net
