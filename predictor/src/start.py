from tasks import *
import sys


def run_with_logs(task_number, executable):
    log("Begin of Task " + str(task_number))
    executable()
    log("End of Task " + str(task_number))


if len(sys.argv) == 1:
    run_with_logs(1, task1)
    run_with_logs(2, task2)
    run_with_logs(3, task3)
    run_with_logs(4, task4)
    run_with_logs(5, task5)
else:
    try:
        task = int(sys.argv[1])
    except:
        raise ValueError("Invalid argument")

    if task == 1:
        run_with_logs(1, task1)
    elif task == 2:
        run_with_logs(2, task2)
    elif task == 3:
        run_with_logs(3, task3)
    elif task == 4:
        run_with_logs(4, task4)
    elif task == 5:
        run_with_logs(5, task5)
    else:
        raise ValueError("Invalid argument")
