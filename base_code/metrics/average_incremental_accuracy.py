from typing import List, Dict

from base_code.metrics.average_accuracy import average_accuracy


def average_incremental_accuracy(
    accuracies: Dict[str, List[float]],
    task_quantity: int,
    task_prefix: str = "Task",
) -> Dict[str, float]:
    """
    Compute the average incremental accuracy of the predictions.

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
        The average incremental accuracy of the predictions.
    """

    average_accuracies_metrics = average_accuracy(
        accuracies=accuracies,
        task_quantity=task_quantity,
        task_prefix=task_prefix,
    )

    average_incremental_accuracies = []

    for k in range(task_quantity):
        average_incremental_accuracies.append(
            sum(average_accuracies_metrics[: k + 1]) / (k + 1)
        )

    return average_incremental_accuracies
