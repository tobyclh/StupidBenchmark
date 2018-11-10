from time import time

def timereps(reps, func):
    start = time()
    for i in range(0, reps):
        func()
    end = time()
    return (end - start) / reps
