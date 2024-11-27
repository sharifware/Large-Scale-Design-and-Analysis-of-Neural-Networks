from typing import Union
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

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
        self.loss_history = []

        # Ensure directories for saving networks exist
        self.working_dir = "Working Networks"
        self.broken_dir = "Broken Networks"
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

    def __train_network__(self, model: nn.Module, train_loader: DataLoader, num_epochs: int, optimizer, loss_fn: torch.nn.Module):
        """
        Trains a single network model over a specified number of epochs with early stopping.

        Args:
            model (nn.Module): The PyTorch neural network model to train.
            train_loader (DataLoader): The DataLoader for the training dataset.
            num_epochs (int): Number of epochs to train the model.
            optimizer (Optimizer): The optimizer used to update model weights.
            loss_fn (torch.nn.Module): The loss function used to compute the training loss.
        """
        model.train()
        best_loss = None
        epochs_no_improve = 0
        early_stopping_patience = None
        early_stopping_min_delta = 0.05

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for batch in train_loader:
                input, target = batch
                optimizer.zero_grad()           
                output = model(input)           
                loss = loss_fn(output, target)  
                loss.backward()                 
                optimizer.step()                

            current_loss = loss.item()
            self.__show_loss__(epoch, current_loss)  # Print loss per epoch
            self.loss_history.append(current_loss)

            # Early Stopping Logic
            if best_loss is None:
                best_loss = current_loss
            elif current_loss < best_loss - early_stopping_min_delta:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            # model.final_loss = loss.item()      # Save the final loss in the model

    def __train_network_GPU__(self, model: nn.Module, train_loader: DataLoader, num_epochs: int, optimizer, loss_fn: torch.nn.Module, device):
        """
        Trains a single network model over a specified number of epochs using GPU

        Args:
            model (nn.Module): The PyTorch neural network model to train.
            train_loader (DataLoader): The DataLoader for the training dataset.
            num_epochs (int): Number of epochs to train the model.
            optimizer (Optimizer): The optimizer used to update model weights.
            loss_fn (torch.nn.Module): The loss function used to compute the training loss.
            device: The GPU to be used. For now we just assume one.
        """
        model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for batch in train_loader:
                input, target = batch
                input, target = input.to(device), target.to(device)
                optimizer.zero_grad()           
                output = model(input)           
                loss = loss_fn(output, target)  
                loss.backward()                 
                optimizer.step()                
            self.__show_loss__(epoch, loss.item())  # Print loss per epoch
            self.loss_history.append(loss.item())    # Save the final loss in the model

    def __check_success__(self, network: torch.nn.Module ):
        print(f"loss_history[-1] = {self.loss_history[-1]}")
        print(f"loss_history[-2] = {self.loss_history[-2]}")

        if self.loss_history[-1] <= self.success_loss and (self.loss_history[-1] - self.loss_history[-2] < self.convergence_threshold):
            self.working_networks[f'network_{self.success_count+1}'] = network.state_dict()
            # torch.save(network.state_dict(), os.path.join(self.working_dir, f'network_{self.attempt+1}.pt'))
            self.attempt = 0
            self.success_count += 1
        else:
            self.broken_networks[f'network_{self.success_count+1}_{self.attempt}'] = network.state_dict()
            self.attempt += 1  # Increment attempt counter
            # torch.save(network.state_dict(), os.path.join(self.broken_dir, f'network_{self.attempt+1}.pt'))
    
    def __save_networks__(self):
        torch.save(self.working_networks, f'{self.working_dir}/working_networks.pt')
        torch.save(self.broken_networks, f'{self.broken_dir}/broken_networks.pt')
        
    
    def generate_networks(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module):
        """
        Generates multiple neural networks and trains them until a successful number of networks is produced,
        based on the success criteria and the maximum number of allowed attempts.

        Args:
            train_loader (DataLoader): The DataLoader for the training dataset.
            test_loader (DataLoader): The DataLoader for the testing dataset (unused here but included for completeness).
            num_epochs (int): The number of epochs for which each network should be trained.
            loss_fn (torch.nn.Module): The loss function to use during training.
        """

        while self.success_count < self.amount_to_produce:
            print(f"Training Network {self.attempt+1}")

            # Stop if maximum attempts are reached
            if self.attempt >= self.max_attempts:
                print("Error: Maximum number of attempts reached without meeting success criteria.")
                break

            #reset loss hist
            self.loss_history = []

            # Instantiate a new network
            network_to_be = self.model_architecture()
            optimizer = torch.optim.SGD(network_to_be.parameters(), lr=0.1, momentum=0.9)  # SGD optimizer
            
            # Train the network
            self.__train_network__(network_to_be, train_loader, num_epochs, optimizer, loss_fn)

            # Check if the network meets the success criteria
            self.__check_success__(network_to_be)
            

        if self.success_count == self.amount_to_produce:
            print(f"Successfully trained {self.success_count} networks.")
        else:
            print(f"{self.success_count} successful networks created out of {self.amount_to_produce}.")
        
        print(f"Attemps:{self.attempt}, successfull networks: {self.success_count}")

        self.__save_networks__()
        
        # Reset these parameters to run the Analyzer again
        self.attempt = 0
        self.success_count = 0
    
    def generate_networks_GPU(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module, learning_rate, momentum_user):
        """
        Generates multiple neural networks and trains them until a successful number of networks is produced,
        based on the success criteria and the maximum number of allowed attempts.

        Args:
            train_loader (DataLoader): The DataLoader for the training dataset.
            test_loader (DataLoader): The DataLoader for the testing dataset (unused here but included for completeness).
            num_epochs (int): The number of epochs for which each network should be trained.
            loss_fn (torch.nn.Module): The loss function to use during training.
        """

        # Check for Nvidia GPU
        if not torch.cuda.is_available():
            print("NVIDIA  GPU not available.")
        else:
            device = torch.device("cuda")
            print(f"Using {device} device!")
        
        # Check for Mac M GPU
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")

        else:
            device = torch.device("mps")
            print(f"Using {device} device!")

        # Start Generating Networks
        while self.success_count < self.amount_to_produce:
            print(f"Training Network {self.success_count+1}")

            # Stop if maximum attempts are reached
            if self.attempt >= self.max_attempts:
                print("Error: Maximum number of attempts reached without meeting success criteria.")
                break

            # Instantiate a new network
            network_to_be = self.model_architecture()
            optimizer = torch.optim.SGD(network_to_be.parameters(), lr=learning_rate, momentum=momentum_user)  # SGD optimizer
            
            # Train the network
            network_to_be.to(device)
            self.__train_network_GPU__(network_to_be, train_loader, num_epochs, optimizer, loss_fn, device)

            # Check if the network meets the success criteria
            self.__check_success__(network_to_be)
            self.attempt += 1  # Increment attempt counter

        # Display amount of netwroks created/attempted
        if self.success_count == self.amount_to_produce:
            print(f"Successfully trained {self.success_count} networks.")
        else:
            print(f"{self.success_count} successful networks created out of {self.amount_to_produce}.")
        
        print(f"Attemps:{self.attempt}, successfull networks: {self.success_count}")
        print("Resetting internal counter of networks...")
        
        # Reset these parameters to run the Analyzer again
        self.attempt = 0
        self.success_count = 0

    def generate_networks_parallel(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs: int, loss_fn: torch.nn.Module):
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
            optimizer = torch.optim.SGD(network_to_be.parameters(), lr=0.1, momentum=0.9)  # SGD optimizer
            
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


        
