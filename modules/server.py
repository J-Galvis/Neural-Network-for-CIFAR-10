import socket
import pickle
import torch
import torch.nn as nn
from defineNetwork import Net

if __name__ == '__main__':
    HOST = 'localhost'
    PORT = 6000

    # Initialize model and optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Import trainloader inside main block to avoid Windows multiprocessing issues
    from defineNetwork import trainloader
    
    # Prepare batches
    batches = list(trainloader)

    # Start server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f'Server listening on {HOST}:{PORT}')

    num_workers = 2  # You can change this
    worker_sockets = []
    for i in range(num_workers):
        conn, addr = server_socket.accept()
        print(f'Worker {i+1} connected from {addr}')
        worker_sockets.append(conn)

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

    def send_batch(sock, batch):
        """Send batch data to worker"""
        try:
            data = pickle.dumps(batch)
            sock.sendall(len(data).to_bytes(4, 'big') + data)
            return True
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
            print(f"Connection error while sending batch: {e}")
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
    for epoch in range(2):  # Training for 2 epochs
        print(f'Epoch {epoch+1}')
        batch_idx = 0
        
        while batch_idx < len(batches):
            batch_gradients = []
            batches_sent = 0
            workers_to_remove = []
            
            # Send batches to active workers
            for i, ws in enumerate(active_workers):
                if batch_idx >= len(batches):
                    # Send termination signal for this round
                    if not send_batch(ws, b'DONE'):
                        workers_to_remove.append(i)
                    break
                
                batch = batches[batch_idx]
                if send_batch(ws, batch):
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
            
            # Receive and aggregate gradients from remaining workers
            successful_gradients = []
            workers_to_remove = []
            
            for i in range(min(batches_sent, len(active_workers))):
                if i < len(active_workers):
                    worker_grads = receive_gradients(active_workers[i])
                    if worker_grads is not None:
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
        
        if not active_workers:
            print("No active workers remaining. Stopping training...")
            break
            
        print(f'Epoch {epoch+1} finished with {len(active_workers)} active workers')

    # Send termination signal to remaining workers
    print("Sending termination signals to remaining workers...")
    for ws in active_workers:
        try:
            ws.sendall(len(b'DONE').to_bytes(4, 'big') + b'DONE')
        except:
            pass
        ws.close()

    server_socket.close()
    print("Server stopped")
