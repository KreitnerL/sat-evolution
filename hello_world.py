import sys
import time
print("Hello World from the Server!")

class ICH(object):
    height = 180

    def __init__(self):
        self.height = 190
        self.width = 100
        print("success")

if __name__ == '__main__':
    a = ICH()
    time.sleep(10)
    print(sys.argv[1])
    raise ValueError("ERROR of DEATH")
    print(a.width)