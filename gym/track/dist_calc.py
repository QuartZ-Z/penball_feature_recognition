import numpy as np

sum = 0
count = 0
while True:
    str = input()
    x1, y1, x2, y2 = map(int, str.split())
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    sum += dist
    count += 1
    print("average dist: ", sum / count)
    print("dist: ", dist)