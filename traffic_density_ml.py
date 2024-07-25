import joblib
import numpy as np

# Load the model
model_path = 'random_forest_model.pkl'
model = joblib.load(model_path)

# Modify the dtype of the node array
for estimator in model.estimators_:
    tree = estimator.tree_
    tree_nodes = tree.__getstate__()['nodes']
    expected_dtype = np.dtype([
        ('left_child', '<i8'),
        ('right_child', '<i8'),
        ('feature', '<i8'),
        ('threshold', '<f8'),
        ('impurity', '<f8'),
        ('n_node_samples', '<i8'),
        ('weighted_n_node_samples', '<f8'),
        ('missing_go_to_left', 'u1')
    ])

    modified_nodes = np.zeros(tree_nodes.shape, dtype=expected_dtype)
    for field in tree_nodes.dtype.names:
        if field in expected_dtype.names:
            modified_nodes[field] = tree_nodes[field]

    tree.__setstate__({'nodes': modified_nodes, **tree.__getstate__()})

# Save the modified model
modified_model_path = 'modified_random_forest_model.pkl'
joblib.dump(model, modified_model_path)
