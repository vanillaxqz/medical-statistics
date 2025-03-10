import matplotlib.pyplot as plt
from collections import Counter


def plot_histogram(chosen, numeric_data, text_data):

    if numeric_data:
        plt.figure()
        plt.hist(numeric_data, bins=10, edgecolor='black')
        plt.title(f"Histogram of {chosen}")
        plt.xlabel(chosen)
        plt.ylabel("Frequency")
        plt.show()
    else:
        counts = Counter(text_data)
        unique_values = list(counts.keys())
        frequencies = list(counts.values())
        plt.figure()
        plt.bar(unique_values, frequencies)
        plt.title(f"Histogram of {chosen} (Categorical)")
        plt.xlabel(chosen)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_box_plot(chosen_label, numeric_data):
    plt.figure()
    plt.boxplot(numeric_data, labels=[f"{chosen_label} (numeric)"])
    plt.title("Box Plot")
    plt.ylabel("Values")
    plt.show()


def plot_grouped_box_plot(numeric_label, grouping_label, grouped_data):
    plt.figure()
    data_to_plot = list(grouped_data.values())
    group_names = list(grouped_data.keys())
    plt.boxplot(data_to_plot, labels=group_names)
    plt.title(f"Box Plot of {numeric_label} grouped by {grouping_label}")
    plt.ylabel("Values")
    plt.show()
