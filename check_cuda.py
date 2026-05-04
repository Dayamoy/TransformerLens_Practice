import torch

def main():
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}\n")
    
    if cuda_available:
        # Get the number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs detected: {gpu_count}")
        
        # Loop through and print the name and capability of each GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("CUDA is not available. PyTorch is using the CPU.")

if __name__ == "__main__":
    main()