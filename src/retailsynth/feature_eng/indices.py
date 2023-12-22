"""Helper class to keep track of the index for feature arrays."""
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class IndexMap(object):
    """helper class to keep track of the index for feature arrays.

    Parameters
    ----------
        values (list): list of values to be indexed
        name (str, optional): name of the index map. Defaults to None.
    """

    values: List
    name: str = field(default="")

    def __post_init__(self):
        """Create a dictionary mapping with {value: integer}."""
        self._index_map = np.argsort(self.values)

    def get_index(self, values: List) -> List:
        """Return the index list for the given values.

        Parameters
        ----------
            values (list): list of values to be indexed

        Returns
        -------
            list: list of indices for the given values
        """
        if not isinstance(values, list):
            values = [values]

        indices = self._index_map[
            np.searchsorted(self.values, values, sorter=self._index_map)
        ].tolist()
        return indices  # type: ignore

    def get_values(self, indices: List) -> List:
        """Return the values list for the given indices.

        Parameters
        ----------
            indices (list): list of indices to be searched for

        Returns
        -------
            list: list of values for the given indices
        """
        if not isinstance(indices, list):
            indices = [indices]

        selected_values = np.array(self.values)[indices].tolist()
        return selected_values  # type: ignore

    def __len__(self) -> int:
        """Return the length of the index map.

        Returns
        -------
            int: length of the index map
        """
        return len(self.values)
