import click
import torch
from typing import Sized, Tuple, cast
from torchvision import datasets, transforms

from pytorch_demos.NMIST.simple_cnn import ConvNet

DEFAULT_OUTPUT_DIRECTORY = "saved_models"


class CNNTrainer:
    """
    CLass to train the CNN
    """

    def __init__(
        self,
        model: ConvNet,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        log_interval: int,
        dry_run: bool,
    ):
        """
        train for one epoch

        :param data_loader: an object holding all the training data
        :param epoch: which epoch this is
        :param log_interval: how often to log results
        :param dry_run: break after one loop if true
        :return:
        """
        self.model.train()
        for batch_idx, data in enumerate(data_loader):
            img: torch.Tensor = cast(torch.Tensor, data[0])
            target: torch.Tensor = cast(torch.Tensor, data[1])
            data, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(cast(Sized, data_loader.dataset)),
                        100.0 * batch_idx / len(data_loader),
                        loss.item(),
                    )
                )
                if dry_run:
                    break

    def test(self, data_loader: torch.utils.data.DataLoader):
        """
        run the model on the test set

        :param data_loader: an object holding all the test data
        :return:
        """
        self.model.eval()
        test_loss: float = 0
        correct: int = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += torch.nn.functional.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                prediction = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        test_loss /= len(cast(Sized, data_loader.dataset))
        return test_loss, correct

    def train_model(
        self,
        training_data: torch.utils.data.Dataset,
        test_data: torch.utils.data.Dataset,
        batch_size: int = 64,
        test_batch_size: int = 1000,
        epochs: int = 14,
        dry_run: bool = False,
        seed: int = 1,
        log_interval: int = 10,
        save_model: bool = False,
        output_directory: str = DEFAULT_OUTPUT_DIRECTORY,
        output_filename: str = "mnist_cnn",
    ):
        """
        function to train the cnn model

        :param training_data: all the training data
        :param test_data: all the test data
        :param batch_size: how big a training batch size to use
        :param test_batch_size: how big a test batch size to use
        :param epochs: how many epochs to train for
        :param dry_run: break after one loop if true
        :param seed: random seed
        :param log_interval: how often to log training results
        :param save_model: set to true to save out the model afterward
        :param output_directory: where to save the model to (if save_model=True)
        :param output_filename: what to save the output model as (if save_model=True)
        :return:
        """
        # set the random seed
        torch.manual_seed(seed)

        # choose device to use
        device = torch.device("cpu")  # train on CPU

        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size)

        # initialize model
        self.model.to(device)

        for epoch in range(epochs):
            self.training_step(train_loader, epoch, log_interval, dry_run)
            test_loss, correct = self.test(test_loader)
            print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(cast(Sized, test_loader.dataset)),
                    100.0 * correct / len(cast(Sized, test_loader.dataset)),
                )
            )
            self.scheduler.step()

        if save_model:
            self.model.save_model(file_name=output_filename, folder=output_directory)


def train_cnn(
    save_model: bool = False,
    output_directory: str = DEFAULT_OUTPUT_DIRECTORY,
    output_filename: str = "mnist_cnn",
    batch_size: int = 64,
    test_batch_size: int = 1000,
    epochs: int = 14,
    learning_rate: float = 1.0,
    gamma: float = 0.7,
    dry_run: bool = False,
    seed: int = 1,
    log_interval: int = 10,
):
    """
    Function to set up and Train a CNN on NMIST

    :param save_model: set to true to save out the model afterward
    :param output_directory: where to save the model to (if save_model=True)
    :param output_filename: what to save the output model as (if save_model=True)
    :param batch_size: how big a training batch size to use
    :param test_batch_size: how big a test batch size to use
    :param epochs: how many epochs to train for
    :param learning_rate: the learning rate
    :param gamma: the gamma rate
    :param dry_run: set to True to break after one loop
    :param seed: the random seed
    :param log_interval: how often to log results
    :return:
    """
    device = torch.device("cpu")  # train on CPU

    # set up dataloaders
    image_mean = 0.13066047627384256  # mean value of the images in our training data
    image_std = (
        0.3050502568821657  # standard deviation of the images in our training data
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((image_mean,), (image_std,))]
    )

    training_data = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        "../data", train=False, download=False, transform=transform
    )

    # initialize model
    model = ConvNet(image_size=(28, 28), output_categories=10).to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    trainer = CNNTrainer(model, device, optimizer, scheduler)
    trainer.train_model(
        training_data=training_data,
        test_data=test_data,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        dry_run=dry_run,
        seed=seed,
        log_interval=log_interval,
        save_model=save_model,
        output_directory=output_directory,
        output_filename=output_filename,
    )


@click.command()
@click.option(
    "--save_model", default=False, type=bool, help="whether you want to save the model"
)
@click.option(
    "--output_directory",
    default=DEFAULT_OUTPUT_DIRECTORY,
    type=str,
    help="where to save the model",
)
@click.option(
    "--output_filename",
    default="mnist_cnn",
    type=str,
    help="what to name the saved model",
)
@click.option("--batch_size", default=64, type=int, help="size of training batches")
@click.option("--test_batch_size", default=1000, type=int, help="size of test batches")
@click.option("--epochs", default=14, type=int, help="number of epochs to train for")
@click.option(
    "--learning_rate", default=1.0, type=float, help="optimizer learning rate"
)
@click.option("--gamma", default=0.7, type=float, help="scheduler gamma value")
@click.option(
    "--dry_run", default=False, type=bool, help="set to True to break after one loop"
)
@click.option("--seed", default=1, type=int, help="random seed")
@click.option("--log_interval", default=10, type=int, help="how often to log progress")
def run_train_cnn(
    save_model: bool = False,
    output_directory: str = DEFAULT_OUTPUT_DIRECTORY,
    output_filename: str = "mnist_cnn",
    batch_size: int = 64,
    test_batch_size: int = 1000,
    epochs: int = 14,
    learning_rate: float = 1.0,
    gamma: float = 0.7,
    dry_run: bool = False,
    seed: int = 1,
    log_interval: int = 10,
):
    train_cnn(
        save_model=save_model,
        output_directory=output_directory,
        output_filename=output_filename,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        dry_run=dry_run,
        seed=seed,
        log_interval=log_interval,
    )


if __name__ == "__main__":
    run_train_cnn()
