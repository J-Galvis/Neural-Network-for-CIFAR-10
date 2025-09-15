import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn
import socket
import pickle
import torch
import time
import csv
import os

from defineNetwork import Net
from testing import testingNetwork

HOST = '192.168.0.137' 
PORT = 6000

def send_model_params(sock, model):
    """Send current model parameters to worker (parameters only)"""
    try:
        params = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
        data = pickle.dumps(params)
        sock.sendall(len(data).to_bytes(4, 'big') + data)
        return True
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
        print(f"Connection error while sending parameters: {e}")
        return False

def send_batch_list(sock, batch_list):
    """Send list of batch IDs to worker for the epoch"""
    try:
        data = pickle.dumps(batch_list)
        sock.sendall(len(data).to_bytes(4, 'big') + data)
        return True
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
        print(f"Connection error while sending batch list: {e}")
        return False

def receive_gradients(sock):
    """Receive accumulated gradients from worker after epoch"""
    try:
        grad_len = int.from_bytes(sock.recv(4), 'big')
        grad_data = b''
        while len(grad_data) < grad_len:
            chunk = sock.recv(min(grad_len - len(grad_data), 4096))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            grad_data += chunk
        return pickle.loads(grad_data)
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
        print(f"Connection error while receiving gradients: {e}")
        return None

def accuracyTest(net, transform, num_workers):
    print("starting testing . . . ")
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=num_workers)
    return testingNetwork(testloader, net)

def start_server(num_workers=2, num_epochs=20, saveFile = './Results/cifar10_trained_model.pth'):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))

    net = Net()

    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-2, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    # Initialize CSV files for time tracking
    results_dir = './Results'
    os.makedirs(results_dir, exist_ok=True)
    
    server_time_file = os.path.join(results_dir, 'Server time.csv')
    net_time_file = os.path.join(results_dir, 'net_Times.csv')
    
    # Initialize Server time CSV
    with open(server_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Total_Epoch_Time', 'Active_Workers'])
    
    # Get the total number of batches
    total_batches = len(trainloader)

    # Start server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f'Server listening on {HOST}:{PORT}')

    worker_sockets = []
    for i in range(num_workers):
        conn, addr = server_socket.accept()
        print(f'Worker {i+1} connected from {addr}')
        worker_sockets.append(conn)

    # Send initial model parameters to all workers
    print("Sending initial model parameters to workers...")
    active_workers = []
    for i, ws in enumerate(worker_sockets):
        if send_model_params(ws, net):
            active_workers.append(ws)
            print(f"Successfully sent parameters to worker {i+1}")
        else:
            print(f"Failed to send parameters to worker {i+1}, removing from active workers")
            ws.close()

    if not active_workers:
        print("No active workers available. Exiting...")
        server_socket.close()
        exit(1)

    print(f"Training with {len(active_workers)} active workers")

    # Training loop
    net_training_start = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch+1}')
        
        # Distribute batches among workers for this epoch
        batches_per_worker = total_batches // len(active_workers)
        remaining_batches = total_batches % len(active_workers)
        
        workers_to_remove = []
        
        # Send batch assignments to each worker
        batch_start = 0
        for i, ws in enumerate(active_workers):
            # Calculate how many batches this worker gets
            worker_batch_count = batches_per_worker + (1 if i < remaining_batches else 0)
            worker_batch_list = list(range(batch_start, batch_start + worker_batch_count))
            batch_start += worker_batch_count
            
            if send_batch_list(ws, worker_batch_list):
                print(f"Sent {len(worker_batch_list)} batches to worker {i+1}")
            else:
                print(f"Failed to send batch list to worker {i+1}, removing from active workers")
                workers_to_remove.append(i)
                ws.close()
        
        # Remove disconnected workers
        for i in reversed(workers_to_remove):
            active_workers.pop(i)
        
        if not active_workers:
            print("All workers disconnected. Stopping training...")
            break
        
        # Wait for all workers to finish their batches and send accumulated gradients
        print(f"Waiting for accumulated gradients from {len(active_workers)} workers...")
        successful_gradients = []
        workers_to_remove = []
        
        for i, ws in enumerate(active_workers):
            worker_grads = receive_gradients(ws)
            if worker_grads is not None:
                successful_gradients.append(worker_grads)
                print(f"Received accumulated gradients from worker {i+1}")
            else:
                print(f"Failed to receive gradients from worker {i+1}")
                workers_to_remove.append(i)
                ws.close()
        
        # Remove workers that failed to send gradients
        for i in reversed(workers_to_remove):
            active_workers.pop(i)
        
        # Average gradients and update model (once per epoch)
        if successful_gradients:
            optimizer.zero_grad()
            
            # Average gradients from all successful workers
            num_workers = len(successful_gradients)
            for param_idx, param in enumerate(net.parameters()):
                if param_idx < len(successful_gradients[0]):
                    # Average gradients from all successful workers for this parameter
                    avg_grad = sum(torch.tensor(grads[param_idx], device=param.device, dtype=param.dtype) 
                                 for grads in successful_gradients) / num_workers
                    param.grad = avg_grad
            
            # Apply gradient clipping before optimizer step
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            # Update model parameters
            optimizer.step()
            scheduler.step()
            
            print(f"Model updated after epoch {epoch+1} using {num_workers} workers")
            
            # Send updated parameters to remaining workers for next epoch
            if epoch < num_epochs - 1:  # Don't send if this is the last epoch
                workers_to_remove = []
                for i, ws in enumerate(active_workers):
                    if not send_model_params(ws, net):
                        print(f"Failed to send updated parameters to worker {i+1}")
                        workers_to_remove.append(i)
                        ws.close()
                
                # Remove workers that couldn't receive updates
                for i in reversed(workers_to_remove):
                    active_workers.pop(i)
        
        # Calculate total epoch time
        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time
        
        # Log server epoch time
        with open(server_time_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{total_epoch_time:.4f}", len(active_workers)])
        
        if not active_workers:
            print("No active workers remaining. Stopping training...")
            break
            
        print(f'Epoch {epoch+1} finished with {len(active_workers)} active workers (Time: {total_epoch_time:.4f}s)')

    # Calculate total net training time
    net_training_total = time.time() - net_training_start

    Accuracy = accuracyTest(net, transform, num_workers)

    # Log net training time
    with open(net_time_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_epochs, f"{net_training_total:.4f}", f"{Accuracy:.2f}"])

    # Send termination signal to remaining workers
    print("Sending termination signals to remaining workers...")
    for ws in active_workers:
        try:
            ws.sendall(len(b'DONE').to_bytes(4, 'big') + b'DONE')
        except:
            pass
        ws.close()

    torch.save(net.state_dict(), saveFile)
    server_socket.close()
    print("Server stopped")

if __name__ == '__main__':
    start_server()