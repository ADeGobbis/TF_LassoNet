from tf_lassonet.model import LassoNet
from typing import Optional, List
from dataclasses import dataclass
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tqdm.auto import tqdm


@dataclass
class HistoryItem:
    lambda_: float
    objective: float  # loss + lambda_ * regulatization
    loss: float
    val_objective: float  # val_loss + lambda_ * regulatization
    val_loss: float
    regularization: float
    n_selected_features: int
    selected_features: np.ndarray
    n_iters: int
    test_predictions: np.ndarray


def compute_feature_importances(path: List[HistoryItem]):
    """When does each feature disappear on the path?
    Parameters
    ----------
    path : List[HistoryItem]
    Returns
    -------
        feature_importances_
    """

    current = path[0].selected_features.copy()
    ans = ans = np.full(current.shape, float("inf"))
    for save in path[1:]:
        lambda_ = save.lambda_
        diff = current & ~save.selected_features
        ans[diff.nonzero()] = lambda_
        current &= save.selected_features
    return ans

class LassoPath:
    def __init__(
        self,
        model,
        n_iters_init: int,
        patience_init: int,
        n_iters_path: int,
        patience_path: int,
        lambda_seq: Optional[List[float]] = None,
        lambda_start: Optional[float] = None,
        path_multiplier: float = 1.02,
        M: float = 10,
        eps_start: float = 1,
        restore_best_weights: bool = False,
    ):
        self.lassonet = LassoNet(model, M=M)

        self.n_iters_init = n_iters_init
        self.patience_init = patience_init
        self.n_iters_path = n_iters_path
        self.patience_path = patience_path
        self.lambda_seq = lambda_seq
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier
        self.eps_start = eps_start
        self.restore_best_weights = restore_best_weights

   
    def fit_one_model(
        self, x, y, val_dataset=None, *, test_dataset=None, lambda_, **kwargs
    ) -> HistoryItem:
        self.lassonet.lambda_.assign(lambda_)

        history = self.lassonet.fit(
            x,
            y,
            validation_data=val_dataset,
            epochs=self.n_iters_init,
            callbacks=[
                EarlyStopping(
                    patience=self.patience_init,
                    restore_best_weights=self.restore_best_weights,
                )
            ],
            verbose=True,
            **kwargs
        )

        reg = self.lassonet.regularization()
        if val_dataset is not None:
            if len(val_dataset) == 2:
                val_loss = self.lassonet.evaluate(val_dataset[0], val_dataset[1])
            else:
                val_loss = self.lassonet.evaluate(val_dataset)
        else:
            val_loss = 0.0
            print('WARNING: loss on test set not defined')

        test_predictions = None
        if test_dataset is not None:            
            test_predictions = self.lassonet.predict(test_dataset)
        return HistoryItem(
            lambda_=lambda_,
            loss=history.history["loss"][-1],
            objective=history.history["loss"][-1] + lambda_ * reg,
            val_loss=val_loss,
            val_objective=val_loss + lambda_ * reg,
            regularization=reg.numpy(),
            n_iters=len(history.history["loss"]),
            n_selected_features=self.lassonet.selected_count().numpy(),
            selected_features=self.lassonet.input_mask().numpy().astype('int32'),
            test_predictions=test_predictions
        )

    def _update_bar(self, i: int, bar, h, lambda_: float):
        bar.update(1)
        bar.set_postfix(
            {
                "Lambda": lambda_,
                "Val loss": h.val_loss,
                "Selected features": h.n_selected_features,
                "Regularization": h.regularization,
            }
        ),

    def fit(
        self, x, y=None, val_dataset = None, verbose: bool = False, **kwargs
    ) -> List[HistoryItem]:
        self.history = []
        if verbose:
            bar = tqdm()
            bar.update(0)

        h = self.fit_one_model(x, y, val_dataset=val_dataset , lambda_=0, **kwargs)
        self.history.append(h)

        if verbose:
            self._update_bar(1, bar, h, 0)
            
         # build lambda_seq
        lambda_seq = self.lambda_seq
        if lambda_seq is None:

            def _lambda_seq(start):
                while True:
                    yield start
                    start *= self.path_multiplier

            if self.lambda_start is not None:
                lambda_seq = _lambda_seq(self.lambda_start)
            else:
                lambda_seq = _lambda_seq(self.eps_start * self.history[-1].loss)


        for i, current_lambda in enumerate(lambda_seq):

            h = self.fit_one_model(
                x, y, val_dataset=val_dataset, lambda_=current_lambda, **kwargs
            )
            self.history.append(h)
            finalize = self.lassonet.selected_count()[0] == 0
            if verbose:
                self._update_bar(i + 2, bar, h, current_lambda)
                if finalize:
                    bar.close()

            if finalize:
                break
        return self.history

    def compute_feature_importances(self):
        """When does each feature disappear on the path?
        Parameters
        ----------
        path : List[HistoryItem]
        Returns
        -------
            feature_importances_
        """
        return compute_feature_importances(self.history)
