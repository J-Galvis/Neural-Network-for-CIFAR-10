import socket
import pickle
import torch
import torch.nn as nn
from defineNetwork import Net

HOST = 'localhost'
PORT = 6000

# Initialize model (parameters will be synced from server)
net = Net()
criterion = nn.CrossEntropyLoss()

# Connect to server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print("Connected to server")

def receive_data(sock):
    """Receive data with length prefix"""
    data_len = int.from_bytes(sock.recv(4), 'big')
    data = b''
    while len(data) < data_len:
        data += sock.recv(data_len - len(data))
    return data

def send_gradients(sock, gradients):
    """Send gradients to server"""
    grad_data = pickle.dumps(gradients)
    sock.sendall(len(grad_data).to_bytes(4, 'big') + grad_data)

def update_model_params(model, params_dict):
    """Update model parameters from server"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in params_dict:
                param.data = torch.tensor(params_dict[name], dtype=param.dtype)

try:
    # Receive initial model parameters
    params_data = receive_data(client_socket)
    initial_params = pickle.loads(params_data)
    update_model_params(net, initial_params)
    print("Received initial model parameters")

    while True:
        # Receive batch or termination signal
        data = receive_data(client_socket)
        
        # Check for termination signal
        if data == b'DONE':
            print("Received termination signal")
            break
        
        # Process batch
        try:
            batch = pickle.loads(data)
            inputs, labels = batch
            
            # Forward pass and compute gradients
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Extract gradients
            grads = [param.grad.cpu().numpy() for param in net.parameters() if param.grad is not None]
            
            # Send gradients back to server
            send_gradients(client_socket, grads)
            
            # Receive updated model parameters
            params_data = receive_data(client_socket)
            if params_data != b'DONE':
                updated_params = pickle.loads(params_data)
                update_model_params(net, updated_params)
            
            print(f"Processed batch, loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            break

except Exception as e:
    print(f"Worker error: {e}")
finally:
    client_socket.close()
    print("Worker disconnected")
