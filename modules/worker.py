import socket
import pickle
import torch
import torch.nn as nn
from defineNetwork import Net
from defineNetwork import trainloader

def receive_data(sock):
    """Receive data with length prefix"""
    try:
        # Receive length prefix with timeout
        sock.settimeout(30.0)  # 30 second timeout
        length_bytes = b''
        while len(length_bytes) < 4:
            chunk = sock.recv(4 - len(length_bytes))
            if not chunk:
                raise ConnectionError("Connection closed while receiving length")
            length_bytes += chunk
        
        data_len = int.from_bytes(length_bytes, 'big')
        
        # Receive actual data
        data = b''
        while len(data) < data_len:
            chunk = sock.recv(min(data_len - len(data), 4096))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            data += chunk
        
        sock.settimeout(None)  # Remove timeout
        return data
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError) as e:
        print(f"Connection error while receiving data: {e}")
        sock.settimeout(None)
        return None

def send_gradients(sock, gradients):
    """Send gradients to server"""
    try:
        grad_data = pickle.dumps(gradients)
        sock.sendall(len(grad_data).to_bytes(4, 'big') + grad_data)
        return True
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError) as e:
        print(f"Connection error while sending gradients: {e}")
        return False

def update_model_params(model, params_dict):
    """Update model parameters from server"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in params_dict:
                param.data = torch.tensor(params_dict[name], dtype=param.dtype)
                
def start_worker():
    HOST = '10.180.208.105'
    PORT = 6000

    # Initialize model (parameters will be synced from server)
    net = Net()
    criterion = nn.CrossEntropyLoss()

    # Prepare batches list to access by index
    batches = list(trainloader)
    print(f"Worker loaded {len(batches)} batches locally")

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print("Connected to server")

    try:
        # Receive initial model parameters
        params_data = receive_data(client_socket)
        if params_data is None:
            print("Failed to receive initial parameters")
            exit(1)
        
        initial_params = pickle.loads(params_data)
        update_model_params(net, initial_params)
        print("Received initial model parameters")

        batch_count = 0
        while True:
            # Receive batch ID or termination signal
            data = receive_data(client_socket)
            
            if data is None:
                print("Connection lost, terminating worker")
                break
            
            # Check for termination signal
            if data == b'DONE':
                print("Received termination signal")
                break
            
            # Process batch ID
            try:
                batch_id = pickle.loads(data)
                
                # Debug: Check what we received
                if isinstance(batch_id, bytes) and batch_id == b'DONE':
                    print("Received DONE signal as batch data")
                    break
                elif not isinstance(batch_id, int):
                    print(f"Unexpected batch ID format: {type(batch_id)}, value: {batch_id}")
                    continue
                
                # Check if batch ID is valid
                if batch_id < 0 or batch_id >= len(batches):
                    print(f"Invalid batch ID: {batch_id}, available range: 0-{len(batches)-1}")
                    continue
                
                # Get the actual batch data using the batch ID
                batch = batches[batch_id]
                inputs, labels = batch[0], batch[1]
                
                # Forward pass and compute gradients
                net.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Extract gradients
                grads = [param.grad.cpu().numpy() for param in net.parameters() if param.grad is not None]
                
                # Send gradients back to server
                if not send_gradients(client_socket, grads):
                    print("Failed to send gradients, terminating")
                    break
                
                # Receive updated model parameters
                params_data = receive_data(client_socket)
                if params_data is None:
                    print("Failed to receive updated parameters, terminating")
                    break
                    
                if params_data != b'DONE':
                    try:
                        updated_params = pickle.loads(params_data)
                        update_model_params(net, updated_params)
                    except Exception as e:
                        print(f"Error updating model parameters: {e}")
                        break
                
                batch_count += 1
                if batch_count % 10 == 0:  # Print every 10 batches to reduce output
                    print(f"Processed {batch_count} batches (last batch ID: {batch_id}), last loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error processing batch ID: {e}")
                print(f"Batch ID data type: {type(batch_id) if 'batch_id' in locals() else 'undefined'}")
                print(f"Raw data length: {len(data) if data else 'None'}")
                break

    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass
        print("Worker disconnected")

if __name__ == "__main__":
    start_worker()