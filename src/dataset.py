import torch
from typing import Any, Union


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset class for classification tasks.

    Parameters:
        encodings: Dictionary containing the input encodings.
        labels: List of labels.
    """

    def __init__(self, encodings: dict[str, Any], labels: Union[list[int], list[str]]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, int, str]]:
        """
        Retrieve an item from the dataset.

        Parameters:
            idx: Index of the item to retrieve.

        Returns:
            Dictionary containing the input encodings and the label.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.labels)
