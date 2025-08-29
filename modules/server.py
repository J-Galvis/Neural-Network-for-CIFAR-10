import socket
import pickle
import torch
import torch.nn as nn
from defineNetwork import Net, trainloader

HOST = 'localhost'
PORT = 6000

# Initialize model and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

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
    params = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
    data = pickle.dumps(params)
    sock.sendall(len(data).to_bytes(4, 'big') + data)

def send_batch(sock, batch):
    """Send batch data to worker"""
    data = pickle.dumps(batch)
    sock.sendall(len(data).to_bytes(4, 'big') + data)

def receive_gradients(sock):
    """Receive gradients from worker"""
    grad_len = int.from_bytes(sock.recv(4), 'big')
    grad_data = b''
    while len(grad_data) < grad_len:
        grad_data += sock.recv(grad_len - len(grad_data))
    return pickle.loads(grad_data)

# Send initial model parameters to all workers
print("Sending initial model parameters to workers...")
for ws in worker_sockets:
    send_model_params(ws, net)

# Training loop
for epoch in range(2):  # Increased to 2 epochs for better testing
    print(f'Epoch {epoch+1}')
    batch_idx = 0
    total_loss = 0.0
    
    while batch_idx < len(batches):
        batch_gradients = []
        batches_sent = 0
        
        # Send batches to workers
        for ws in worker_sockets:
            if batch_idx >= len(batches):
                # Send termination signal
                ws.sendall(len(b'DONE').to_bytes(4, 'big') + b'DONE')
                break
            batch = batches[batch_idx]
            send_batch(ws, batch)
            batch_idx += 1
            batches_sent += 1
        
        # Receive and aggregate gradients from workers
        for i in range(batches_sent):
            worker_grads = receive_gradients(worker_sockets[i])
            batch_gradients.append(worker_grads)
        
        # Average gradients and apply to model
        if batch_gradients:
            optimizer.zero_grad()
            for param_idx, param in enumerate(net.parameters()):
                # Average gradients from all workers for this parameter
                avg_grad = sum(grads[param_idx] for grads in batch_gradients) / len(batch_gradients)
                param.grad = torch.tensor(avg_grad, dtype=param.dtype)
            
            optimizer.step()
            
            # Send updated parameters to workers
            for ws in worker_sockets:
                send_model_params(ws, net)
    
    print(f'Epoch {epoch+1} finished')

# Send termination signal to all workers
print("Sending termination signals...")
for ws in worker_sockets:
    try:
        ws.sendall(len(b'DONE').to_bytes(4, 'big') + b'DONE')
    except:
        pass
    ws.close()

server_socket.close()
print("Server stopped")
