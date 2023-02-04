class CLS:
    def __init__(self, x, y):
        X = x
        Y = y

#单例模式
def single(cls, *args ,**kw) :
    instances = {}
    def getInstance() :
        if cls not in instances :
            instances[cls]=cls(*args ,**kw)
            print('instances: ', instances)
            print(cls)
            return instances[cls]
    return getInstance


singleton = single(CLS, "abc", "def")
singleton()


