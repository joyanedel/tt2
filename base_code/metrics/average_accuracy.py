from typing import List, Dict


def average_accuracy(
    accuracies: Dict[str, List[float]],
    task_quantity: int,
    task_prefix: str = "Task",
) -> Dict[str, float]:
    """
    Compute the average accuracy of the predictions.

    Parameters
    ----------
    predictions : Dict[str, List[float]]
        The predictions of the model.
    task_quantity : int
        The number of tasks.
    task_prefix : str, optional
        The prefix of the tasks, by default "Task".

    Returns
    -------
    Dict[str, float]
        The average accuracy of the predictions.
    """

    average_accuracies = []

    for k in range(task_quantity):
        av_acc = 0.0
        for j in range(k + 1):
            task_label_j = f"{task_prefix}{j}"
            av_acc += accuracies[task_label_j][k]

        av_acc /= k + 1
        average_accuracies.append(av_acc)

    return average_accuracies
