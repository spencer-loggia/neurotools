import numpy as np
import pandas as pd


class StratifiedSampler:
    """
    Class to create multiple folds where items in the catagorical "strat_cols" and target col are maximally balanced in
    each fold.
    """
    def __init__(self, full_df, n_folds, target_col, stratify=True, strat_cols=()):
        """
        creates a StratifiedSampler object with n_folds folds.
        Parameters
        ----------
        full_df: Full dataframe of stimulus and targets.
        n_folds: number of cross validation folds
        target_col: col with class labels
        strat_cols: other cols that should have values evenly distributed.
        """
        self.og_df = full_df
        self.n_folds = n_folds
        self.target_col = target_col
        self.strat_col = strat_cols + [self.target_col]
        if stratify:
            self.folds = [pd.DataFrame(columns=full_df.columns) for _ in range(n_folds)]
            self._stratified_split_balanced()
        else:
            # full_df = full_df.sample(frac=1.0)
            self.folds = [full_df.iloc[i::n_folds] for i in range(n_folds)]

    def _stratified_split_balanced(self,):
        df = self.og_df
        strat_cols = self.strat_col
        k = self.n_folds

        # Group the DataFrame by the specified stratification columns
        grouped = df.groupby(strat_cols)

        # Loop over each group and distribute rows in a round-robin manner
        counter = 0  # Counter to keep track of DataFrame index in round-robin
        for _, group in grouped:
            for _, row in group.iterrows():
                self.folds[counter % k] = pd.concat([self.folds[counter % k], pd.DataFrame([row])])
                counter += 1

    def get_train(self, idx):
        data = pd.concat(self.folds[:idx] + self.folds[idx + 1:])
        return data

    def get_test(self, idx):
        return self.folds[idx]

    def get_all(self):
        data = pd.concat(self.folds)
        return data