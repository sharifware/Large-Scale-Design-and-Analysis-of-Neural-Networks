
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class NetworkAnalyzer:

    def __init__(self, model_architecture: nn.Module, amount_to_produce: int, success_loss: float, convergence_threshold: float = None, max_attempts: float = None):
        """
        Initializes the NetworkAnalyzer class with the required model architecture and configuration.

        Args:
            model_architecture (nn.Module): A PyTorch neural network model class specifying the architecture.
            amount_to_produce (int): Number of good neural networks to generate based on success criteria.
            success_loss (float): Loss amount that qualifies a good network. 
            convergence_threshold (float, optional): Maximum allowable difference in loss between the last two epochs 
                                            of training to determine if the neural network has converged.
            max_attempts (float, optional): Maximum number of attempts to produce a successful network, defaults
                                            to 130% of `amount_to_produce` if not specified.
        """
        self.model_architecture = model_architecture
        self.amount_to_produce = amount_to_produce
        self.success_loss = success_loss if success_loss is not None else self.__default_success_criteria__()
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else 1000000 #Making it a huge number so it always works
        self.max_attempts = max_attempts if max_attempts is not None else self.__default_max_attempts__()
        
        # Intialize these values to keep track of networks attempts
        self.attempt = 0  
        self.success_count = 0  

        # Array to keep track of loss
        self.train_loss_history = []
        self.test_loss_history = []

        # Ensure directories for saving networks exist
        self.working_dir = "WorkingNetworks"
        self.broken_dir = "BrokenNetworks"
        self.working_networks = {}
        self.broken_networks = {}

        # Set up directories
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)  # Create directory if it doesn't exist
        if not os.path.exists(self.broken_dir):
            os.makedirs(self.broken_dir)

    def __default_success_criteria__(self):
        """
        Provides a default success criteria if none is specified by the user.
        The default is a loss value of 1.0.

        Returns:
            float: The default success criteria (loss value).
        """
        return 1.0

    def __default_max_attempts__(self):
        """
        Sets the maximum number of attempts to produce successful networks to 130% of the amount requested,
        if not specified by the user.

        Returns:
            int: The calculated maximum number of attempts.
        """
        return int(self.amount_to_produce * 1.3)

    def __show_loss__(self, epoch: int, loss_value: float):
        """
        Prints the loss value for a specific epoch during training.

        Args:
            epoch (int): The current epoch number during training.
            loss_value (float): The loss value for the current epoch.
        """
        print(f"Epoch [{epoch+1}], Loss: {loss_value:.4f}")

    def __test_network__(self, model: nn.Module, test_loader: DataLoader, loss_fn: torch.nn.Module, device):
        """
        Tests the network using the testing dataset, and prints the Test loss

        Args:
            model (nn.Module): The trained PyTorch model to evaluate.
            test_loader (DataLoader): DataLoader for the test dataset, providing batches of test data.
            loss_fn (nn.Module): The loss function used to compute the loss (e.g., nn.CrossEntropyLoss, nn.MSELoss).
            device (torch.device): The device on which to perform the evaluation (e.g., "cuda" or "cpu").
        """
        model.eval()
        test_loss = 0.0

        print("Testing for the epoch")

        with torch.no_grad():  # Disable gradient calculation for testing
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()


        # Average loss across all test samples
        avg_loss = test_loss / len(test_loader)
        self.test_loss_history.append(avg_loss)
        print(f"Test Loss: {avg_loss:.4f}")

    def __train_network__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, optimizer, loss_fn: torch.nn.Module, device, test: bool):
        """
        Trains a single network model over a specified number of epochs with early stopping.

        Args:
            model (nn.Module): The PyTorch neural network model to train.
            train_loader (DataLoader): The DataLoader for the training dataset.
            test_loader (DataLoader): The DataLoader for the testing dataset.
            num_epochs (int): Number of epochs to train the model.
            optimizer (Optimizer): The optimizer used to update model weights.
            loss_fn (torch.nn.Module): The loss function used to compute the training loss.
            device: The GPU to be used. For now, we just assume one.
            test (bool): Whether to test the model after each epoch.
        """

        model.to(device)
        for epoch in range(num_epochs):
            avg_loss = self.__train_one_epoch__(model, train_loader, optimizer, loss_fn, device, epoch)
            self.train_loss_history.append(avg_loss)

            if test:
                self.__test_network__(model, test_loader, loss_fn, device)

    def __train_one_epoch__(self, model: nn.Module, train_loader: DataLoader, optimizer, loss_fn: torch.nn.Module, device, epoch: int) -> float:
        """
        Trains the model for one epoch and returns the average loss.

        Args:
            model (nn.Module): The PyTorch neural network model.
            train_loader (DataLoader): DataLoader for training data.
            optimizer: The optimizer for updating model weights.
            loss_fn (torch.nn.Module): Loss function for training.
            device: The GPU to be used.
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        model.train()
        running_loss = 0.0

        for input, target in train_loader:
            input, target = input.to(device), target.to(device)
            
            optimizer.zero_grad()
            loss = self.__compute_loss__(model, input, target, loss_fn)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        self.__show_loss__(epoch, avg_loss)

        return avg_loss

    def __compute_loss__(self, model: nn.Module, input, target, loss_fn: torch.nn.Module) -> torch.Tensor:
        """
        Computes the loss for a single batch.

        Args:
            model (nn.Module): The PyTorch model.
            input: The input tensor.
            target: The target tensor.
            loss_fn (torch.nn.Module): The loss function.

        Returns:
            torch.Tensor: The computed loss value.
        """
        output = model(input)
        return loss_fn(output, target)

    def __check_success__(self, network: torch.nn.Module):
        """
        Checks if the trained network meets the success criteria.
        Stores successful and broken networks accordingly.
        """
        if self.__is_successful_network__():
            self.__store_successful_network__(network)
        else:
            self.__store_broken_network__(network)

    def __is_successful_network__(self) -> bool:
        """
        Determines if the network meets success criteria.
        """
        return (
            self.train_loss_history[-1] <= self.success_loss and
            (self.train_loss_history[-1] - self.train_loss_history[-2] < self.convergence_threshold)
        )

    def __store_successful_network__(self, network: torch.nn.Module):
        """
        Stores the network as a successful model.
        """
        self.working_networks[f'network_{self.success_count+1}'] = network.state_dict()
        self.success_count += 1

    def __store_broken_network__(self, network: torch.nn.Module):
        """
        Stores the network as a failed attempt.
        """
        self.broken_networks[f'network_{self.success_count+1}_{self.attempt}'] = network.state_dict()
        self.attempt += 1

    def __get_device__(self):
        # Check for Nvidia GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using {device} device!")
        # Check for Mac M GPU
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using {device} device!")
        else:
            device = torch.device("cpu")
            print("No GPU device available. Using CPU.")
        return device
    
    def __save_networks__(self):
        torch.save(self.working_networks, f'{self.working_dir}/working_networks.pt')
        torch.save(self.broken_networks, f'{self.broken_dir}/broken_networks.pt')

    def __visualize_loss__(self, train, test):
        """"
        Uses all the losses while training to plot a graph.
        can only be used after network generation has been
        """
        epochs = range(1, max(len(train), len(test)) + 1)

        # Plot training loss if available
        if train:
            plt.plot(epochs, train, label="Training Loss", marker="o")

        # Plot testing loss if available
        if test:
            plt.plot(epochs, test, label="Testing Loss", marker="o")

        # Add labels, title, and legend
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training vs Testing Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def check_loss(self,  train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module, learning_rate):
        """
        Trains a single network and prints the training and testing loss

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the testing dataset.
            num_epochs (int): The number of epochs to train the network.
            loss_fn (torch.nn.Module): The loss function to calculate the loss during training.
            learning_rate (float): The learning rate for the SGD optimizer.
        """
        device = self.__get_device__()

        network_to_be = self.model_architecture()
        optimizer = torch.optim.SGD(network_to_be.parameters(), lr=learning_rate)  # SGD optimizer

        # Train the network
        network_to_be.to(device)
        self.__train_network__(network_to_be, train_loader, test_loader, num_epochs, optimizer, loss_fn, device, True)

        # Visualize the loss for the single network trained
        self.__visualize_loss__(self.train_loss_history, self.test_loss_history)

        self.__reset_loss__()

    def __reset_loss__(self):
        # Resets the loss history for the next networks
        self.train_loss_history.clear()
        self.test_loss_history.clear()
    
    def generate_networks(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module, learning_rate: float):
        """
        Generates multiple neural networks and trains them until a successful number of networks is produced,
        based on the success criteria and the maximum number of allowed attempts.

        Args:
            train_loader (DataLoader): The DataLoader for the training dataset.
            test_loader (DataLoader): The DataLoader for the testing dataset.
            num_epochs (int): The number of epochs for which each network should be trained.
            loss_fn (torch.nn.Module): The loss function to use during training.
            learning_rate (float): Learning rate for the optimizer.
        """

        device = self.__get_device__()

        while self.__should_continue_training__():
            print(f"Training Network {self.success_count+1}")

            if self.__max_attempts_reached__():
                break

            network_to_be = self.__initialize_network__(learning_rate)
            self.__train_network__(network_to_be, train_loader, test_loader, num_epochs, network_to_be.optimizer, loss_fn, device, False)
            self.__check_success__(network_to_be)

        self.__display_training_summary__()
        self.__save_networks__()
        self.__reset_training_parameters__()

    def __should_continue_training__(self) -> bool:
        """Checks if more networks need to be generated."""
        return self.success_count < self.amount_to_produce

    def __max_attempts_reached__(self) -> bool:
        """Checks if the maximum number of attempts has been reached."""
        if self.attempt >= self.max_attempts:
            print("Error: Maximum number of attempts reached without meeting success criteria.")
            return True
        return False

    def __initialize_network__(self, learning_rate: float) -> nn.Module:
        """Initializes a new neural network and its optimizer."""
        network = self.model_architecture()
        network.optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
        return network

    def __display_training_summary__(self):
        """Displays the final training summary."""
        if self.success_count == self.amount_to_produce:
            print(f"Successfully trained {self.success_count} networks.")
        else:
            print(f"{self.success_count} successful networks created out of {self.amount_to_produce}.")
        
        print(f"Attempts: {self.attempt}, Successful Networks: {self.success_count}")

    def __reset_training_parameters__(self):
        """Resets training parameters to allow rerunning the analyzer."""
        self.attempt = 0
        self.success_count = 0
    
    def generate_networks_parallel(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module, learning_rate):
        """
        Generates multiple neural networks and trains them, *using GPU and in Parallel*, until a successful number of networks is produced,
        based on the success criteria and the maximum number of allowed attempts.

        Args:
            train_loader (DataLoader): The DataLoader for the training dataset.
            test_loader (DataLoader): The DataLoader for the testing dataset (unused here but included for completeness).
            num_epochs (int): The number of epochs for which each network should be trained.
            loss_fn (torch.nn.Module): The loss function to use during training.
        """
         
        # Check for Nvidia GPU, exit if no GPU
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA GPU not available. Cannot train in Parallel")
        else:
            device = torch.device("cuda")
            print(f"Using {device} device!")

        while self.success_count < self.amount_to_produce:
            print(f"Training Network {self.attempt+1}")

            # Stop if maximum attempts are reached
            if self.attempt >= self.max_attempts:
                print("Error: Maximum number of attempts reached without meeting success criteria.")
                break

            # Instantiate a new network
            network_to_be = self.model_architecture()
            optimizer = torch.optim.SGD(network_to_be.parameters(), lr=learning_rate)  # SGD optimizer
            
            # Train the network
            network_to_be = torch.nn.DataParallel(network_to_be)
            self.__train_network_parallel__(network_to_be, train_loader, num_epochs, optimizer, loss_fn, device)

            # Check if the network meets the success criteria
            self.__check_success__(network_to_be)
            self.attempt += 1  # Increment attempt counter

        if self.success_count == self.amount_to_produce:
            print(f"Successfully trained {self.success_count} networks.")
        else:
            print(f"{self.success_count} successful networks created out of {self.amount_to_produce}.")
        
        print(f"Attemps:{self.attempt}, successfull networks: {self.success_count}")
        print("Resetting internal counter of networks...")
        
        # Reset these parameters to run the Analyzer again
        self.attempt = 0
        self.success_count = 0
