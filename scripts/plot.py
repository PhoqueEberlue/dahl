import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def load_training_outputs(path):
    results = pd.read_csv(path, sep=",")
    return list(results['accuracy']), list(results['loss'])

def load_alumet_results(path):
    alumet_results = pd.read_csv(path, sep=";")
    energy_results = alumet_results[alumet_results['metric'].str.contains("rapl_consumed_energy")]

    cumulative_energy = []
    total_energy = 0

    for energy in energy_results["value"]:
        total_energy += energy
        cumulative_energy.append(total_energy)

    timestamps = []

    for timestamp in energy_results["timestamp"]:
        # Strip last characters because datetime does not support more than 6 digits for seconds
        timestamps.append(datetime.strptime(timestamp.rstrip("Z")[:26], "%Y-%m-%dT%H:%M:%S.%f"))

    start_training_time = timestamps[0]

    for i, time in enumerate(timestamps):
        timestamps[i] = (time - start_training_time).seconds

    return cumulative_energy, timestamps

def plot_training(py_energy, py_timestamps, py_accuracy, py_loss, 
                  da_energy, da_timestamps, da_accuracy, da_loss, 
                  name):

    plt.figure(figsize=(12, 5))

    # Plot energy
    plt.subplot(1, 3, 1)
    plt.plot(py_timestamps, py_energy, label=f"Pytorch energy")
    plt.plot(da_timestamps, da_energy, label=f"DAHL energy")
    plt.xlabel("Training time")
    plt.ylabel("Energy in Joules")
    plt.title("Estimated training consumption of the process in Joules")
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(list(range(len(py_loss))), py_loss, label=f"Pytorch loss")
    plt.plot(list(range(len(da_loss))), da_loss, label=f"DAHL loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(list(range(len(py_loss))), py_accuracy, label=f"Pytorch accuracy")
    plt.plot(list(range(len(da_loss))), da_accuracy, label=f"DAHL accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle(name)
    plt.savefig(f"{name}.png")
    plt.show()

da_energy, da_timestamp = load_alumet_results("../build/alumet-output.csv")
da_acc, da_loss = load_training_outputs("../build/dahl-training-outputs.csv")

py_energy, py_timestamp = load_alumet_results("../python-version/alumet-output.csv")
py_acc, py_loss = load_training_outputs("../python-version/pytorch-training-output.csv")

print(da_acc)
print(py_acc)
plot_training(py_energy, py_timestamp, py_acc, py_loss, 
              da_energy, da_timestamp, da_acc, da_loss,
              "Epoch: 20, Samples: 6000, NCPU: 8, Batch size: 10")
