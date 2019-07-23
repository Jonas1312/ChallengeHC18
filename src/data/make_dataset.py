def generate_test_valid_indices():
    """To be used with torch.utils.data.Subset"""
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split

    output_dir = "../../data/processed/"

    X = np.arange(999)

    X_left, X_test, y_left, _ = train_test_split(X, np.ones_like(X), test_size=0.15)
    X_train, X_valid, _, _ = train_test_split(
        X_left, y_left, test_size=0.05 / (1 - 0.15)
    )

    train_indices = X_train
    test_indices = X_test
    valid_indices = X_valid

    assert len(train_indices) + len(test_indices) + len(valid_indices) == 999

    assert not set(train_indices) & set(test_indices)
    assert not set(test_indices) & set(valid_indices)
    assert not set(train_indices) & set(valid_indices)

    np.save(os.path.join(output_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
    np.save(os.path.join(output_dir, "valid_indices.npy"), valid_indices)


if __name__ == "__main__":
    to_run = (generate_test_valid_indices,)
    for func in to_run:
        ret = input(f'Run "{func.__name__}"? (y/n) ')
        if "y" in ret:
            func()
            break
