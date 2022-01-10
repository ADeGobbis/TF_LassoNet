from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from .path import HistoryItem


def feature_importance_histogram(
    feature_importances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    horizontal: bool = True,
    N:Optional[int] = None,
    ax=None,
    **kwargs
):
    if ax is None:
        _, ax = plt.subplots(**kwargs)
    importance_indices = np.argsort(feature_importances)
    x_ticks = []
    labels = []
    if N is None:
        N = feature_importances.shape[0]
    for i, j in enumerate(importance_indices[::-1][:N][::-1]):
        if feature_names is not None:
            labels.append(feature_names[j])
        if horizontal:
            ax.barh(
                y=i,
                width=feature_importances[j],
                height=1,
                edgecolor="#4477DD",
                facecolor="blue",
            )
        else:
            ax.bar(
                x=i,
                height=feature_importances[j],
                width=1,
                edgecolor="#4477DD",
                facecolor="blue",
            )
        x_ticks.append(i)
    if feature_names is not None:
        if horizontal:
            ax.set_yticks(x_ticks)
            ax.set_yticklabels(labels)
        else:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(labels)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    return ax

def plot_history(*args: List[HistoryItem], figsize = (8, 8), title = None, legend = None):

    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score


    Parameters
    ==========
    path: the result of a LassoPath.fit()
    """
    n_selected = []
    score = []
    lambda_ = []
    for path in args:
        n_cur = []
        score_cur = []
        lambda_cur = []
        for history in path:
            n_cur.append(history.n_selected_features[0])
            score_cur.append(history.val_loss)
            lambda_cur.append(history.lambda_)
        n_selected.append(n_cur)
        score.append(score_cur)
        lambda_.append(lambda_cur)


    plt.figure(figsize=figsize)

    plt.subplot(311)
    plt.grid(True)
    for x, y in zip(n_selected, score):
        plt.plot(x, y, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("score")
    if legend is not None:
        plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplot(312)
    plt.grid(True)
    for x, y in zip(lambda_, score):
        plt.plot(x, y, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")
    if legend is not None:
        plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplot(313)
    plt.grid(True)
    for x, y in zip(lambda_, n_selected):
        plt.plot(x, y, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")
    if legend is not None:
        plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))

    if title is not None:
        plt.title(title)

    plt.tight_layout()
