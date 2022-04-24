import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.5720399403745379
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=True, l1_ratio=0.5, learning_rate="constant", loss="perceptron", penalty="elasticnet", power_t=0.5)),
    GradientBoostingClassifier(learning_rate=0.001, max_depth=8, max_features=0.7500000000000001, min_samples_leaf=18, min_samples_split=2, n_estimators=100, subsample=0.25)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
