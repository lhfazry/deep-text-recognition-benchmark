import argparse
import re
import matplotlib.pyplot as plt
import os

def parse_logs(log_path):
    iterations = []
    train_losses = []
    valid_losses = []
    accuracies = []

    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return None

    # Regex patterns for matching log entries
    # Sample format: [iteration/total] Train loss: X, Valid loss: Y, Elapsed_time: Z
    # Current_accuracy: A, Current_norm_ED: B
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
        for i in range(len(lines)):
            line = lines[i]
            # Match Training/Validation loss line
            loss_match = re.search(r'\[(\d+)/(\d+)\] Train loss: ([\d\.]+), Valid loss: ([\d\.]+)', line)
            if loss_match:
                iterations.append(int(loss_match.group(1)))
                train_losses.append(float(loss_match.group(3)))
                valid_losses.append(float(loss_match.group(4)))
                
                # Check next line for accuracy
                if i + 1 < len(lines):
                    acc_match = re.search(r'Current_accuracy\s+:\s+([\d\.]+)', lines[i+1])
                    if acc_match:
                        accuracies.append(float(acc_match.group(1)))
                    else:
                        # Sometimes accuracy might be on subsequent lines or formatted differently
                        # Fallback for search in nearby lines
                        for j in range(1, 4):
                            if i + j < len(lines):
                                acc_match = re.search(r'Current_accuracy\s+:\s+([\d\.]+)', lines[i+j])
                                if acc_match:
                                    accuracies.append(float(acc_match.group(1)))
                                    break

    return iterations, train_losses, valid_losses, accuracies

def plot_metrics(iterations, train_losses, valid_losses, accuracies, output_dir, show_plots=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, train_losses, label='Train Loss')
    plt.plot(iterations, valid_losses, label='Valid Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    # Plot Accuracy
    if len(accuracies) == len(iterations):
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, accuracies, label='Accuracy', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        acc_plot_path = os.path.join(output_dir, 'accuracy_plot.png')
        plt.savefig(acc_plot_path)
        print(f"Accuracy plot saved to {acc_plot_path}")
    else:
        print("Warning: Count of iterations and accuracies don't match. Skipping accuracy plot.")
    
    if show_plots:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training logs')
    parser.add_argument('--log_path', type=str, required=True, help='Path to log_train.txt')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--show', action='store_true', help='Show plots immediately')
    
    args = parser.parse_args()
    
    data = parse_logs(args.log_path)
    if data:
        iterations, train_losses, valid_losses, accuracies = data
        if iterations:
            plot_metrics(iterations, train_losses, valid_losses, accuracies, args.output_dir, show_plots=args.show)
        else:
            print("No data found in log file.")
