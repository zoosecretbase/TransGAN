class MyClass():
    def __init__(self) -> None:
        self.attribute = 1
    
    def say_hello(self):
        return 'helloooooooo'

if __name__ == '__main__':
    m = MyClass()
    print(m.attribute)
    print(m.say_hello())