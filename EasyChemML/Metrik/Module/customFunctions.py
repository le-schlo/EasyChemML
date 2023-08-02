import numpy as np
def HighestDens(a: np.ndarray, num_highest: int) -> np.ndarray:
    """
    Problem: Bei gleichen Werten wird der letztere genommen
    """

    idx_dens = np.array([np.argsort(a)[-(i + 1)] for i in range(num_highest)])

    return idx_dens

# def CompareHighestDens(self, y_true, y_pred, num_highest=5) -> float:
#     self.topk = num_highest
#     true_idx = self.HighestDens(y_true, num_highest=num_highest)
#     pred_idx = self.HighestDens(y_pred, num_highest=num_highest)
#
#     error = 0
#     for i, pos in enumerate(true_idx):
#         if pos != pred_idx[i]:
#             error += 1
#             if pred_idx[i] in true_idx:
#                 error -= 0.3
#
#     return error / len(y_pred)
def ranked_score(y_true, y_pred, num_highest=10) -> float:
    # true_idx = self.HighestDens(self.true_Dens, num_highest=num_highest)
    # pred_idx = self.HighestDens(self.pred_Dens, num_highest=num_highest)

    if len(y_true) < num_highest:
        num_highest = len(y_true)

    if len(y_pred) < num_highest:
        num_highest = len(y_pred)

    ha = HighestDens(y_true, num_highest=num_highest)
    hb = HighestDens(y_pred, num_highest=num_highest)

    take_into_account = int(num_highest / 2)

    error = 0

    for i, posi in enumerate(ha[:take_into_account]):
        if posi in hb:
            error += np.abs(list(hb).index(posi) - i)
        else:
            error += num_highest

    for i, posi in enumerate(hb[:take_into_account]):
        if posi in ha:
            error += np.abs(list(ha).index(posi) - i)
        else:
            error += num_highest

    return error / (2 * take_into_account)