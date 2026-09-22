"""Group crops that show the same person.

Two crops taken from the same photo are always two different people, because
stage 1 already removed the duplicates within a photo. That constraint stops
the obvious failure of this approach, which is merging two supporters who are
standing next to each other in the same coat.
"""

import numpy as np


class _Union:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def merge(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def cluster(
    vectors: np.ndarray, sources: list[str], threshold: float
) -> list[list[int]]:
    """Group row indices of `vectors` into profiles, closest pairs first.

    `sources` names the photo each row came from. Clusters are returned
    largest first, and every row appears in exactly one of them.
    """
    count = len(vectors)
    if count == 0:
        return []

    similarity = vectors @ vectors.T
    rows, cols = np.triu_indices(count, k=1)
    close = similarity[rows, cols] >= threshold
    rows, cols = rows[close], cols[close]
    order = np.argsort(-similarity[rows, cols])

    union = _Union(count)
    photos = {index: {sources[index]} for index in range(count)}
    for pair in order:
        a, b = union.find(int(rows[pair])), union.find(int(cols[pair]))
        if a == b or photos[a] & photos[b]:
            continue
        union.merge(a, b)
        photos[union.find(a)] = photos[a] | photos[b]

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(union.find(index), []).append(index)
    return sorted(grouped.values(), key=lambda members: (-len(members), members[0]))
