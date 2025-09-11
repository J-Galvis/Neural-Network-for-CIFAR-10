import socket
import pickle
import torch
import time
import torch.nn as nn
import csv
import os
from defineNetwork import Net
from defineNetwork import trainloader

def send_model_params(sock, model):
    """Send current model parameters to worker"""
    try:
        params = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
        data = pickle.dumps(params)
        sock.sendall(len(data).to_bytes(4, 'big') + data)
        return True
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
        print(f"Connection error while sending parameters: {e}")
        return False

def send_batch_id(sock, batch_id):
    """Send batch ID to worker"""
    try:
        data = pickle.dumps(batch_id)
        sock.sendall(len(data).to_bytes(4, 'big') + data)
        return True
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
        print(f"Connection error while sending batch ID: {e}")
        return False

def receive_gradients(sock):
    """Receive gradients from worker"""
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

def start_server(num_workers=2):

    HOST = 'localhost' 
    PORT = 6000

    # Initialize model and optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Initialize CSV files for time tracking
    results_dir = './Results'
    os.makedirs(results_dir, exist_ok=True)
    
    workers_time_file = os.path.join(results_dir, 'Workers time.csv')
    server_time_file = os.path.join(results_dir, 'Server time.csv')
    
    # Initialize Workers time CSV
    with open(workers_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Worker_ID', 'Batch_Processing_Time', 'Total_Worker_Time_Per_Epoch'])
    
    # Initialize Server time CSV
    with open(server_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Total_Epoch_Time', 'Active_Workers'])
    
    # Get the total number of batches (we only need the count now)
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
    for epoch in range(5):  # Training for 5 epochs
        epoch_start_time = time.time()
        worker_times = {}  # Track time for each worker in this epoch
        
        print(f'Epoch {epoch+1}')
        batch_idx = 0
        
        # Reset batch_idx for each epoch
        while batch_idx < total_batches:
            batch_gradients = []
            batches_sent = 0
            workers_to_remove = []
            batch_start_times = {}  # Track when workers
            
            # Send batches to active workers, but only send as many batches as we have workers
            num_batches_to_send = min(len(active_workers), total_batches - batch_idx)
            
            for i in range(num_batches_to_send):
                if i < len(active_workers):
                    ws = active_workers[i]
                    batch_id = batch_idx  # Send the batch index as ID
                    
                    batch_start_times[i] = time.time()  
                    if send_batch_id(ws, batch_id):
                        batch_idx += 1
                        batches_sent += 1
                    else:
                        print(f"Worker {i+1} disconnected, removing from active workers")
                        workers_to_remove.append(i)
                        ws.close()
            
            # Remove disconnected workers
            for i in reversed(workers_to_remove):
                active_workers.pop(i)
            
            if not active_workers:
                print("All workers disconnected. Stopping training...")
                break
            
            # Only try to receive gradients for the batches we actually sent
            successful_gradients = []
            workers_to_remove = []
            
            for i in range(batches_sent):
                if i < len(active_workers):
                    worker_grads = receive_gradients(active_workers[i])
                    if worker_grads is not None:
                        # Calculate batch processing time for this worker
                        batch_end_time = time.time()
                        batch_processing_time = batch_end_time - batch_start_times[i]
                        
                        # Track worker time for this epoch
                        if i not in worker_times:
                            worker_times[i] = 0
                        worker_times[i] += batch_processing_time
                        
                        # Log individual batch processing time
                        with open(workers_time_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([epoch+1, i+1, f"{batch_processing_time:.4f}", ""])
                        
                        successful_gradients.append(worker_grads)
                    else:
                        print(f"Failed to receive gradients from worker {i+1}")
                        workers_to_remove.append(i)
                        active_workers[i].close()
            
            # Remove workers that failed to send gradients
            for i in reversed(workers_to_remove):
                if i < len(active_workers):
                    active_workers.pop(i)
            
            # Average gradients and apply to model
            if successful_gradients:
                optimizer.zero_grad()
                for param_idx, param in enumerate(net.parameters()):
                    if param_idx < len(successful_gradients[0]):
                        # Average gradients from all successful workers for this parameter
                        avg_grad = sum(grads[param_idx] for grads in successful_gradients) / len(successful_gradients)
                        param.grad = torch.tensor(avg_grad, dtype=param.dtype)
                
                optimizer.step()
                
                # Send updated parameters to remaining workers
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
        
        # Log total worker times for this epoch
        for worker_id, total_time in worker_times.items():
            with open(workers_time_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, worker_id+1, "", f"{total_time:.4f}"])
        
        # Log server epoch time
        with open(server_time_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{total_epoch_time:.4f}", len(active_workers)])
        
        if not active_workers:
            print("No active workers remaining. Stopping training...")
            break
            
        print(f'Epoch {epoch+1} finished with {len(active_workers)} active workers (Time: {total_epoch_time:.4f}s)')

    # Send termination signal to remaining workers
    print("Sending termination signals to remaining workers...")
    for ws in active_workers:
        try:
            ws.sendall(len(b'DONE').to_bytes(4, 'big') + b'DONE')
        except:
            pass
        ws.close()
    

    torch.save(net.state_dict(), './Results/cifar10_trained_model.pth') #This saves the trained model

    server_socket.close()
    print("Server stopped")


if __name__ == '__main__':
    start_server(1)