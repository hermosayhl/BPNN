import numpy
import datetime


def add_dim(x, dim=0):
	return numpy.expand_dims(x, axis=dim)


def one_hot(x, _len):
    if(hasattr(x, "__len__")):
        target = numpy.zeros((len(x), _len))
        for i, it in enumerate(x):
        	target[i, int(it)] = 1
        return target
    else:
        target = numpy.zeros((_len))
        target[int(x)] = 1
    return target


class Timer:
    def __enter__(self):
        self.start = datetime.datetime.now()

    def __exit__(self, type, value, trace):
        _end = datetime.datetime.now()
        print('耗时  :  {}'.format(_end - self.start))



def clear_cache():
    import os
    import platform
    sys_name = platform.system()
    if(sys_name == 'Windows'):
        os.system("rd /s ./__pycache__/")
    elif(sys_name == 'Linux'):
        os.system("rm -rf ./__pycache__/")




def shuffle_togather(arrs):
    state = numpy.random.get_state()
    for it in arrs:
        numpy.random.set_state(state)
        numpy.random.shuffle(it)
    return tuple(arrs)


if __name__ == '__main__':
    lhs = [1, 2, 4]
    rhs = [-1, -2, -4]
    shuffle_togather([lhs, rhs])
    print(lhs, rhs)