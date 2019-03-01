import numpy as np
import sys
if __name__ == "__main__":
    data = np.load(sys.argv[1])
    print(np.max(data), "max value")
    print(data.shape, "shape")
    print(data[-10:], "last 10")
