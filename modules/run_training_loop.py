import os
import time
from defineNetwork import trainNet

def run_training_loop(i: int):

    print(f"Starting training loop from 0 to {i} epochs...")

    os.makedirs('./Results', exist_ok=True)

    experiment_start_time = time.time()

    for i in range(1, i + 1):
        print(f"\n{'='*50}")
        print(f"TRAINING RUN {i}: Training network with {i} epochs")
        print(f"{'='*50}")
        
        try:
            trainNet(i)
            print(f"Completed training run {i} with {i} epochs")
            
        except Exception as e:
            print(f"Error during training run {i}: {str(e)}")
            continue
        
        print(f"Training run {i} finished successfully!\n")

if __name__ == "__main__":
    run_training_loop(10)
