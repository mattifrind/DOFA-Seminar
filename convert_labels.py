import numpy as np

conversionMap = [
    [0, 1],
    [2],
    [11, 12, 13],
    [14, 15, 16, 18],
    [17],
    [19],
    [20],
    [21],
    [22],
    [23],
    [24],
    [25, 31],
    [26, 27],
    [28],
    [29],
    [33, 34],
    [35, 36],
    [38, 39],
    [40, 41, 42]
]


def convert_labels(labels):
    new_labels = np.zeros((labels.shape[0], len(conversionMap)))
    for i in range(len(conversionMap)):
        for j in conversionMap[i]:
            new_labels[:, i] += labels[:, j]
    new_labels = np.clip(new_labels, 0, 1)
    return new_labels
