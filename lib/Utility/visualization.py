from sklearn import manifold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def args_to_tensorboard(writer, args):

    txt = ""
    for arg in vars(args):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"

    writer.add_text('command_line_parameters', txt, 0)


def tsne(X, label):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    plt.scatter(np.arange(2), np.arange(2))
    color = ['#1F77B4', '#ED7D31', '#EF6570', '#70AD47', '#BD82BE', '#D1BA74', '#E6CEAC', '#34495e', '#7f8c8d', '#ff7979']
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], '.', color=color[label[i]],
                          fontdict={'size': 66})
    plt.xticks([])
    plt.yticks([])
    plt.show()
