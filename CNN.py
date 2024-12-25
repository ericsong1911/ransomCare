import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
INPUT_LENGTH = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition ---
class KeyFinderCNN(nn.Module):
    def __init__(self, input_length):
        super(KeyFinderCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1) # Or a higher number of outputs for classification.

    def forward(self, x):
      # Add an extra channel dimension to represent multiple channels, in this case it's 1
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# --- Data Loading ---
class MemoryDumpDataset(Dataset):
    def __init__(self, dataframe, input_length):
        self.dataframe = dataframe
        self.input_length = input_length

    def __len__(self):
      return len(self.dataframe)

    def __getitem__(self, idx):
      row = self.dataframe.iloc[idx]
      memory_dump_file = row['memory_dump_file']
      key_address = row['key_address']

      # Find the key address
      if key_address.startswith("["):
         # if there are multiple addresses, take the first
        key_address_list = eval(key_address)
        if key_address_list:
          key_address = int(key_address_list[0])
        else:
          key_address = 0
      else:
        key_address = int(key_address)

      try:
        with open(memory_dump_file, 'rb') as f:
          mem_dump = f.read()
      except FileNotFoundError:
        print(f"File not found: {memory_dump_file}")
        return None, None
      
      mem_dump_length = len(mem_dump)
      
      # Make sure we can load an input window starting at a random offset, and including the key.
      max_start_offset = max(0, min(mem_dump_length-self.input_length, key_address))
      min_start_offset = max(0, key_address-self.input_length+1)
      start_offset = random.randint(min_start_offset, max_start_offset)
      
      # Extract the segment based on input_length, fill with zeros if needed.
      segment = np.frombuffer(mem_dump[start_offset:min(start_offset+self.input_length, mem_dump_length)], dtype=np.uint8)
      if len(segment) < self.input_length:
         padding = np.zeros(self.input_length-len(segment), dtype=np.uint8)
         segment = np.concatenate((segment, padding))
        
      # Convert data to pytorch tensor
      segment = torch.tensor(segment, dtype=torch.float32)
      # Regression output, normalize offset to be between 0 and 1.
      key_offset = torch.tensor((key_address-start_offset)/max(mem_dump_length, 1), dtype = torch.float32)

      return segment, key_offset
   
# --- Training and Evaluation ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
   for epoch in range(num_epochs):
      model.train()
      train_loss = 0.0
      for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        if inputs is None:
           continue
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      
      train_loss /= len(train_loader)
      val_loss = evaluate_model(model, val_loader, criterion, device)

      print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

def evaluate_model(model, val_loader, criterion, device):
   model.eval()
   val_loss = 0.0
   with torch.no_grad():
      for inputs, targets in val_loader:
          if inputs is None:
             continue
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = model(inputs)
          loss = criterion(outputs.squeeze(), targets)
          val_loss += loss.item()
   val_loss /= len(val_loader)
   return val_loss

# --- Main ---
if __name__ == '__main__':
    # Load the database
    df = pd.read_csv("training_data.csv")
    # Split data into training and validation set.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create dataset and dataloader
    train_dataset = MemoryDumpDataset(train_df, INPUT_LENGTH)
    val_dataset = MemoryDumpDataset(val_df, INPUT_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = KeyFinderCNN(INPUT_LENGTH).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)

    # Save the model
    torch.save(model.state_dict(), "key_finder_cnn_model.pth")
    print("Finished training and model saved as key_finder_cnn_model.pth")
