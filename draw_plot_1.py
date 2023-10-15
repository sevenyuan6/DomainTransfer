import umap
import numpy as np
import matplotlib.pyplot as plt


main_cat = [
            '#FF0000',  # red
            '#008000',  # green
            '#0000FF',  # blue
            'purple',  # darkviolet
            ]
markers = ["o", "^", ",", "*"]


def draa(ax, data, label, pp):
    for i in range(len(data)):
        if i % 100 == 0:
            print(f"{i}/{len(data)}")
        main_color = main_cat[label[i]]
        marker = markers[label[i]]
        ax.scatter([data[i][0]], [data[i][1]], color=main_color, zorder=5, s=150, marker=marker)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Layer " + str(pp), fontsize=30)


def plot_embedding(data1, data2, data3, data4, label):
    plt.figure(figsize=(15, 15))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 3

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    draa(ax1, data1, label, 1)
    draa(ax2, data2, label, 2)
    draa(ax3, data3, label, 3)
    draa(ax4, data4, label, 4)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    plt.savefig("first_image.pdf")
    plt.show()


def norm(results):
    x_min, x_max = np.min(results, 0), np.max(results, 0)
    results = (results - x_min) / (x_max - x_min)
    return results


def domain():
    features_1 = np.load("plot_records/feat1.npy")
    features_2 = np.load("plot_records/feat2.npy")
    features_3 = np.load("plot_records/feat3.npy")
    features_4 = np.load("plot_records/feat4.npy")
    labels = np.load("plot_records/label.npy")

    # ts = TSNE(n_components=2, init="pca", random_state=0)
    # results = ts.fit_transform(features)
    umap_model = umap.UMAP(n_components=2, random_state=0)  # You can change n_components to 3 for 3D visualization
    results_1 = umap_model.fit_transform(features_1)
    results_2 = umap_model.fit_transform(features_2)
    results_3 = umap_model.fit_transform(features_3)
    results_4 = umap_model.fit_transform(features_4)

    results_1 = norm(results_1)
    results_2 = norm(results_2)
    results_3 = norm(results_3)
    results_4 = norm(results_4)

    plot_embedding(results_1, results_2, results_3, results_4, labels)


if __name__ == '__main__':
    domain()
