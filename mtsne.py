from MulticoreTSNE import MulticoreTSNE
from sklearn.base import BaseEstimator, TransformerMixin

class MTSNE(BaseEstimator, TransformerMixin):
    def __init__(self,  n_components=2,
                        perplexity=30.0,
                        early_exaggeration=12,
                        learning_rate=200,
                        n_iter=1000,
                        n_iter_without_progress=30,
                        min_grad_norm=1e-07,
                        metric='euclidean',
                        init='random',
                        verbose=0,
                        random_state=None,
                        method='barnes_hut',
                        angle=0.5,
                        n_jobs=1,
                        cheat_metric=True):
        self.set_params(n_components=n_components,
                        perplexity=perplexity,
                        early_exaggeration=early_exaggeration,
                        learning_rate=learning_rate,
                        n_iter=n_iter,
                        n_iter_without_progress=n_iter_without_progress,
                        min_grad_norm=min_grad_norm,
                        metric=metric,
                        init=init,
                        verbose=verbose,
                        random_state=random_state,
                        method=method,
                        angle=angle,
                        n_jobs=n_jobs,
                        cheat_metric=cheat_metric)
    
    def set_params(self, n_components=2,
                   perplexity=30.0,
                   early_exaggeration=12,
                   learning_rate=200,
                   n_iter=1000,
                   n_iter_without_progress=30,
                   min_grad_norm=1e-07,
                   metric='euclidean',
                   init='random',
                   verbose=0,
                   random_state=None,
                   method='barnes_hut',
                   angle=0.5,
                   n_jobs=1,
                   cheat_metric=True):
        self.tsne = MulticoreTSNE(n_components=n_components,
                                  perplexity=perplexity,
                                  early_exaggeration=early_exaggeration,
                                  learning_rate=learning_rate,
                                  n_iter=n_iter,
                                  n_iter_without_progress=n_iter_without_progress,
                                  min_grad_norm=min_grad_norm,
                                  metric=metric,
                                  init=init,
                                  verbose=verbose,
                                  random_state=random_state,
                                  method=method,
                                  angle=angle,
                                  n_jobs=n_jobs,
                                  cheat_metric=cheat_metric)

    def fit_transform(self, X, y=None):
        return self.tsne.fit_transform(X)
