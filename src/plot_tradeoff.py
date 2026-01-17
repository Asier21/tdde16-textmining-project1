import matplotlib.pyplot as plt

def plot_tradeoff():
    # Datos (los tuyos reales)
    methods = ["TF-IDF", "fastText avg", "SBERT"]
    vector_time = [14.566, 61.074, 1291.760]  # segundos
    macro_f1 = [0.9215, 0.8849, 0.8969]

    plt.figure(figsize=(6, 4))

    for x, y, label in zip(vector_time, macro_f1, methods):
        plt.scatter(x, y)
        plt.text(x * 1.05, y, label, fontsize=9, verticalalignment="center")

    plt.xscale("log")  # CLAVE: escala log para que SBERT no aplaste todo
    plt.xlabel("Vectorization time (seconds, log scale)")
    plt.ylabel("Macro-F1 score")
    plt.title("Performance vs Computational Cost Trade-off")

    plt.tight_layout()
    plt.savefig("results/fig_tradeoff.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_tradeoff()
