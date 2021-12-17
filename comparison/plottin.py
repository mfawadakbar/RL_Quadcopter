import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smoothing(data, window_size=15):
    n = len(data)
    average = []

    for index in range(n - window_size):
        average.append(np.mean(data[index : index + window_size]))

    return average


def plot_data(data, x_label, y_label, title):
    """
    Plots the data
    """
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def main():
    """
    Main function
    """

    files = [
        "./comparison/dqn_data.txt",
        "./comparison/ddqn_data.txt",
        "./comparison/a2c_data.txt",
        "./comparison/a3c_data.txt",
    ]

    for file in files:
        data = pd.read_csv(file, header=None)
        time = list(data[3].to_numpy())
        scores = list(data[2].to_numpy())
        window_size = 11
        if file == "./comparison/a3c_data.txt":
            window_size = 15
        average = smoothing(scores, window_size=window_size)
        plt.plot(time[window_size : len(time)], average, alpha=1.0)

    for file in files:
        data = pd.read_csv(file, header=None)
        time = list(data[3].to_numpy())
        scores = list(data[2].to_numpy())
        plt.plot(time, scores, alpha=0.15)

    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.title("Comparison of learning algorithms on Quad-Copter Hover")
    plt.legend(["DQN", "DDQN", "A2C", "A3C"], loc="upper left")
    plt.savefig("./comparison/comparisons.png")


if __name__ == "__main__":
    main()
