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

from defineNetwork import Net, TRANSFORM, NUM_WORKERS, NUM_EPOCHS, SAVE_FILE, TRAINLOADER, PORT, HOST, ImageNetDataset, load_dataset

def testingNetwork(testloader, net):
    """Test network accuracy on validation set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    
    correct = 0
    total = 0
    top5_correct = 0  # Track top-5 accuracy for ImageNet
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Top-5 accuracy for ImageNet
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            # Progress logging for large validation set
            if batch_idx % 50 == 0:
                print(f"Validation batch {batch_idx+1}/{len(testloader)}")
    
    top1_acc = 100 * correct / total
    top5_acc = 100 * top5_correct / total
    print(f"Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%")
    return top1_acc, top5_acc

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
    """Test accuracy on ImageNet validation set"""
    print("Starting ImageNet validation testing...")
    # Load ImageNet validation set
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("ILSVRC/imagenet-1k")
    testset = ImageNetDataset(ds["validation"], transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, 
                                            num_workers=0, pin_memory=False)
    
    return testingNetwork(testloader, net)

def start_server():
    num_workers = NUM_WORKERS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Server using device: {device}")

    net = Net().to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-2, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(TRAINLOADER),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    # Initialize CSV files for time tracking
    results_dir = './Results'
    os.makedirs(results_dir, exist_ok=True)
    
    server_time_file = os.path.join(results_dir, 'Server time.csv')
    
    # Write CSV header if file doesn't exist
    if not os.path.exists(server_time_file):
        with open(server_time_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Epoch_Time_Seconds', 'Total_Training_Time_Seconds', 
                           'Active_Workers', 'Top1_Accuracy', 'Top5_Accuracy'])
    
    # Get the total number of batches
    total_batches = len(TRAINLOADER)

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

    for epoch in range(NUM_EPOCHS):
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
            
            print(f"ImageNet model updated after epoch {epoch+1} using {num_workers} workers")
            print(f"Average learning rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Send updated parameters to remaining workers for next epoch
            if epoch < NUM_EPOCHS - 1:  # Don't send if this is the last epoch
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
        total_epoch_time = time.time() - epoch_start_time
        epoch_training_total = time.time() - net_training_start

        # Run accuracy test (returns top-1 and top-5 for ImageNet)
        top1_acc, top5_acc = accuracyTest(net, TRANSFORM, num_workers)
        
        # Log server epoch time with both accuracy metrics
        with open(server_time_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{total_epoch_time:.4f}", f"{epoch_training_total:.4f}", 
                           len(active_workers), f"{top1_acc:.4f}", f"{top5_acc:.4f}"])
        
        if not active_workers:
            print("No active workers remaining. Stopping training...")
            break
        
        # Clear GPU cache after each epoch for ImageNet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f'ImageNet Epoch {epoch+1} finished with {len(active_workers)} active workers (Time: {total_epoch_time:.4f}s)')
        print(f'Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%')

    # Send termination signal to remaining workers
    print("Sending termination signals to remaining workers...")
    for ws in active_workers:
        try:
            ws.sendall(len(b'DONE').to_bytes(4, 'big') + b'DONE')
        except:
            pass
        ws.close()

    # Save the trained ImageNet model
    torch.save(net.state_dict(), SAVE_FILE)
    print(f"ImageNet trained model saved to: {SAVE_FILE}")
    
    server_socket.close()
    print("ImageNet training server stopped")

if __name__ == '__main__':
    start_server()