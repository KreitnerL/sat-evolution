print("Hello World from the Server!")

class ICH(object):
    height = 180

    def __init__(self):
        self.height = 190
        self.width = 100
        print("success")

if __name__ == '__main__':
    a = ICH()
    print(a.width)