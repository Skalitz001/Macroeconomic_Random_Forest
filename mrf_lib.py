import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

class MRFNode:
    def __init__(self, depth, max_depth, min_obs, mtry, ridge_lambda, podium_zeta):
        self.depth = depth
        self.max_depth = max_depth
        self.min_obs = min_obs
        self.mtry = mtry
        self.ridge_lambda = ridge_lambda
        self.podium_zeta = podium_zeta
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.beta = None
        self.is_leaf = False

    def _get_podium_weights(self, indices, n_total):
        # Implementation of Eq (2): Time-localized weighting
        weights = np.zeros(n_total)
        weights[indices] = 1.0
        
        if self.podium_zeta > 0:
            for t in indices:
                for lag in [-1, 1, -2, 2]:
                    if 0 <= t + lag < n_total:
                        decay = self.podium_zeta if abs(lag)==1 else self.podium_zeta**2
                        weights[t + lag] = max(weights[t + lag], decay)
        
        active_indices = np.where(weights > 0)[0]
        return active_indices, weights[active_indices]

    def fit_ridge(self, X, y, indices, n_total):
        # Fits y_t = X_t * beta_t
        aug_idx, aug_weights = self._get_podium_weights(indices, n_total)
        X_sub = X[aug_idx]
        y_sub = y[aug_idx]
        
        model = Ridge(alpha=self.ridge_lambda, fit_intercept=False) 
        model.fit(X_sub, y_sub, sample_weight=aug_weights)
        
        # Calculate loss (weighted SSE)
        preds = model.predict(X_sub)
        loss = np.sum(aug_weights * (y_sub - preds)**2)
        return model.coef_, loss

    def find_best_split(self, X_linear, S_state, y, indices):
        n_total = len(y)
        best_loss = float('inf')
        best_split = None
        
        # Baseline loss
        _, current_loss = self.fit_ridge(X_linear, y, indices, n_total)
        
        # Random feature selection (Decorrelation)
        n_features = S_state.shape[1]
        feature_indices = np.random.choice(n_features, size=min(self.mtry, n_features), replace=False)
        
        for feat_idx in feature_indices:
            values = np.unique(S_state[indices, feat_idx])
            if len(values) < 5: continue
            
            # Subsample thresholds for speed
            thresholds = np.percentile(values, np.linspace(10, 90, num=10))
            
            for threshold in thresholds:
                left_mask = S_state[indices, feat_idx] <= threshold
                idx_left = indices[left_mask]
                idx_right = indices[~left_mask]
                
                if len(idx_left) < self.min_obs or len(idx_right) < self.min_obs:
                    continue
                
                _, l_loss = self.fit_ridge(X_linear, y, idx_left, n_total)
                _, r_loss = self.fit_ridge(X_linear, y, idx_right, n_total)
                
                if (l_loss + r_loss) < best_loss:
                    best_loss = (l_loss + r_loss)
                    best_split = (feat_idx, threshold, idx_left, idx_right)
                    
        return best_split

    def train(self, X_linear, S_state, y, indices):
        self.beta, _ = self.fit_ridge(X_linear, y, indices, len(y))
        
        if self.depth >= self.max_depth or len(indices) < 2 * self.min_obs:
            self.is_leaf = True
            return

        split = self.find_best_split(X_linear, S_state, y, indices)
        if split is None:
            self.is_leaf = True
            return
            
        self.split_feature, self.split_value, idx_left, idx_right = split
        
        self.left = MRFNode(self.depth + 1, self.max_depth, self.min_obs, self.mtry, self.ridge_lambda, self.podium_zeta)
        self.left.train(X_linear, S_state, y, idx_left)
        
        self.right = MRFNode(self.depth + 1, self.max_depth, self.min_obs, self.mtry, self.ridge_lambda, self.podium_zeta)
        self.right.train(X_linear, S_state, y, idx_right)

    def get_beta(self, s_t_row):
        if self.is_leaf: return self.beta
        if s_t_row[self.split_feature] <= self.split_value:
            return self.left.get_beta(s_t_row)
        return self.right.get_beta(s_t_row)

class MacroRandomForest:
    def __init__(self, n_estimators=50, max_depth=5, min_node_size=15, 
                 ridge_lambda=0.5, podium_zeta=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.ridge_lambda = ridge_lambda
        self.podium_zeta = podium_zeta
        self.trees = []
        
    def fit(self, X_linear, S_state, y):
        # Parallel training
        self.trees = Parallel(n_jobs=-1)(
            delayed(self._train_tree)(X_linear, S_state, y) 
            for _ in range(self.n_estimators)
        )
        return self

    def _train_tree(self, X, S, y):
        # Bagging: Use 70% of data for each tree
        indices = np.random.choice(len(y), int(len(y)*0.7), replace=True)
        mtry = max(1, int(S.shape[1] / 3))
        tree = MRFNode(0, self.max_depth, self.min_node_size, mtry, self.ridge_lambda, self.podium_zeta)
        tree.train(X, S, y, indices)
        return tree

    def predict_gtvps(self, S_state):
        # Returns the Time-Varying Betas
        S_state = np.array(S_state)
        n_obs = len(S_state)
        # Dummy call to get shape
        n_params = len(self.trees[0].get_beta(S_state[0]))
        
        avg_betas = np.zeros((n_obs, n_params))
        for i in range(n_obs):
            betas = np.array([t.get_beta(S_state[i]) for t in self.trees])
            avg_betas[i] = np.mean(betas, axis=0)
        return avg_betas

    def predict(self, X_linear, S_state):
        betas = self.predict_gtvps(S_state)
        return np.sum(X_linear * betas, axis=1)