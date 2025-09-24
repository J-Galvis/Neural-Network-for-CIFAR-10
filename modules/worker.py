import torch.optim as optim
import torch.nn as nn
import socket
import pickle
import torch

from defineNetwork import Net, TRAINLOADER, HOST, PORT

def receive_data(sock):
    """Receive data with length prefix"""
    try:
        sock.settimeout(120.0)  # Longer timeout for epoch processing
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
        
        sock.settimeout(None)
        return data
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError) as e:
        print(f"Connection error while receiving data: {e}")
        sock.settimeout(None)
        return None

def send_gradients(sock, gradients):
    """Send accumulated gradients to server"""
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
                param.data = torch.tensor(params_dict[name], dtype=param.dtype, device=param.device)
    return

def accumulate_gradients(accumulated_grads, current_grads):
    """Accumulate gradients from current batch"""
    if accumulated_grads is None:
        return [grad.clone() for grad in current_grads]
    else:
        for i, grad in enumerate(current_grads):
            accumulated_grads[i] += grad
        return accumulated_grads
                
def start_worker():

    # Initialize model and criterion (no optimizer/scheduler - workers only compute gradients)
    net = Net()
    criterion = nn.CrossEntropyLoss()

    # Prepare batches list to access by index
    batches = list(TRAINLOADER)
    print(f"Worker loaded {len(batches)} batches locally")

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print("Connected to server")

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    
    # Create a dummy optimizer for scaler.unscale_ (required for AMP)
    if scaler is not None:
        dummy_optimizer = optim.SGD(net.parameters(), lr=0.01)

    try:
        # Receive initial model parameters
        params_data = receive_data(client_socket)
        if params_data is None:
            print("Failed to receive initial parameters")
            return
        
        initial_params = pickle.loads(params_data)
        update_model_params(net, initial_params)
        print("Received and applied initial model parameters")

        epoch_count = 0
        
        while True:
            # Receive batch list, updated parameters, or termination signal
            data = receive_data(client_socket)
            
            if data is None:
                print("Connection lost, terminating worker")
                break
            
            # Check for termination signal
            if data == b'DONE':
                print("Received termination signal")
                break
            
            try:
                received_data = pickle.loads(data)
                
                # Check if it's updated parameters (dict) or batch list (list)
                if isinstance(received_data, dict):
                    # This is updated model parameters
                    update_model_params(net, received_data)
                    print("Received and applied updated model parameters")
                    continue
                    
                elif isinstance(received_data, list):
                    # This is a list of batch IDs for this epoch
                    batch_ids = received_data
                    epoch_count += 1
                    print(f"Starting epoch {epoch_count}, processing {len(batch_ids)} batches")
                    
                    # Set model to training mode
                    net.train()
                    
                    # Initialize gradient accumulator
                    accumulated_grads = None
                    total_loss = 0.0
                    
                    # Process all batches for this epoch (gradient computation only)
                    for batch_idx, batch_id in enumerate(batch_ids):
                        if batch_id >= len(batches):
                            print(f"Warning: batch_id {batch_id} exceeds available batches ({len(batches)})")
                            continue
                            
                        inputs, labels = batches[batch_id]
                        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                        # Zero gradients for this batch
                        net.zero_grad()
                        
                        # Forward pass and loss computation
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                outputs = net(inputs)
                                loss = criterion(outputs, labels)
                            
                            # Backward pass to compute gradients
                            scaler.scale(loss).backward()
                            
                            # Unscale gradients for accumulation
                            scaler.unscale_(dummy_optimizer)
                            
                            # Get current gradients (unscaled)
                            current_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) 
                                           for param in net.parameters()]
                        else:
                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            
                            # Backward pass to compute gradients
                            loss.backward()
                            
                            # Get current gradients
                            current_grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) 
                                           for param in net.parameters()]

                        total_loss += loss.item()
                        
                        # Accumulate gradients
                        accumulated_grads = accumulate_gradients(accumulated_grads, current_grads)

                    print(f"Epoch {epoch_count} completed. Avg loss: {total_loss/len(batch_ids):.4f}")
                    
                    # Convert accumulated gradients to numpy and send to server
                    if accumulated_grads:
                        final_grads = [grad.cpu().numpy() for grad in accumulated_grads]
                        
                        if not send_gradients(client_socket, final_grads):
                            print("Failed to send accumulated gradients, terminating")
                            break
                        print(f"Sent accumulated gradients for epoch {epoch_count}")
                        
                    else:
                        print("No gradients to send")
                        break
                
                else:
                    print(f"Unexpected data type received: {type(received_data)}")
                    continue
                
            except Exception as e:
                print(f"Error processing received data: {e}")
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