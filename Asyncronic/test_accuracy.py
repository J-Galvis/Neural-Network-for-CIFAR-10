#!/usr/bin/env python3
"""
Test accuracy script for trained ImageNet neural network
This script loads a trained model and evaluates it on the ImageNet validation set
"""

import torch
import torch.nn as nn
import time
import argparse
import os
from defineNetwork import Net, TRANSFORM, ImageNetDataset, load_dataset

def load_trained_model(model_path, device):
    """Load the trained model from checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize network
    net = Net()
    
    # Load state dict
    if model_path.endswith('.pth'):
        if 'checkpoint' in model_path:
            # Load from checkpoint format
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Load from simple state dict format
            net.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded model state dict")
    else:
        raise ValueError("Model file must be a .pth file")
    
    net.to(device)
    net.eval()
    return net

def test_model_accuracy(net, device, batch_size=32, max_batches=None):
    """Test the model accuracy on ImageNet validation set"""
    print("Loading ImageNet validation dataset...")
    
    # Load ImageNet validation set
    try:
        ds = load_dataset("ILSVRC/imagenet-1k")
        testset = ImageNetDataset(ds["validation"], transform=TRANSFORM)
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Keep at 0 to avoid memory issues
            pin_memory=False
        )
        print(f"Loaded validation set with {len(testset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you're logged into HuggingFace: huggingface-cli login")
        return None, None, None
    
    # Initialize counters
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    batch_times = []
    
    print(f"Starting accuracy test on {len(testloader)} batches...")
    if max_batches:
        print(f"Limited to first {max_batches} batches for quick testing")
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            batch_start = time.time()
            
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = net(images)
            
            # Calculate top-1 accuracy
            _, predicted_top1 = torch.max(outputs, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()
            
            # Calculate top-5 accuracy
            _, predicted_top5 = outputs.topk(5, 1, True, True)
            correct_top5 += predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5)).sum().item()
            
            total += labels.size(0)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress reporting
            if (batch_idx + 1) % 10 == 0:
                current_top1 = 100.0 * correct_top1 / total
                current_top5 = 100.0 * correct_top5 / total
                avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
                print(f"Batch {batch_idx + 1}/{len(testloader)} | "
                      f"Top-1: {current_top1:.2f}% | Top-5: {current_top5:.2f}% | "
                      f"Avg time: {avg_batch_time:.3f}s/batch")
            
            # Early stopping for quick tests
            if max_batches and batch_idx + 1 >= max_batches:
                break
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate final accuracies
    top1_accuracy = 100.0 * correct_top1 / total
    top5_accuracy = 100.0 * correct_top5 / total
    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    return top1_accuracy, top5_accuracy, {
        'total_samples': total,
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'batches_processed': len(batch_times)
    }

def main():
    parser = argparse.ArgumentParser(description='Test ImageNet model accuracy')
    parser.add_argument('--model', type=str, default='./Results/imagenet_trained_model.pth',
                        help='Path to trained model file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for testing (default: 32)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to test (for quick testing)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Available model files:")
        results_dir = './Results'
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith('.pth'):
                    print(f"  - {os.path.join(results_dir, file)}")
        else:
            print("  No Results directory found")
        return
    
    try:
        # Load model
        print(f"Loading model from: {args.model}")
        net = load_trained_model(args.model, device)
        
        # Test accuracy
        top1_acc, top5_acc, stats = test_model_accuracy(
            net, device, args.batch_size, args.max_batches
        )
        
        if top1_acc is not None:
            # Print results
            print("\n" + "="*60)
            print("ACCURACY TEST RESULTS")
            print("="*60)
            print(f"Model: {args.model}")
            print(f"Device: {device}")
            print(f"Samples tested: {stats['total_samples']:,}")
            print(f"Batches processed: {stats['batches_processed']}")
            print(f"Total time: {stats['total_time']:.2f} seconds")
            print(f"Average batch time: {stats['avg_batch_time']:.3f} seconds")
            print("-"*60)
            print(f"Top-1 Accuracy: {top1_acc:.4f}%")
            print(f"Top-5 Accuracy: {top5_acc:.4f}%")
            print("="*60)
            
            # Save results to file
            results_file = args.model.replace('.pth', '_accuracy_results.txt')
            with open(results_file, 'w') as f:
                f.write(f"Model: {args.model}\n")
                f.write(f"Device: {device}\n")
                f.write(f"Samples tested: {stats['total_samples']:,}\n")
                f.write(f"Top-1 Accuracy: {top1_acc:.4f}%\n")
                f.write(f"Top-5 Accuracy: {top5_acc:.4f}%\n")
                f.write(f"Test time: {stats['total_time']:.2f} seconds\n")
            
            print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()