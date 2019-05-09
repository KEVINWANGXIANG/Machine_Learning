import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        #yiled类似于return，rstrip()删除字符串末尾的指定字符(默认为空格)
        yield line.rstrip()

#sys.stdin为标准化输入的方法
input = read_input(sys.stdin)
input = [float(line) for line in input]
print(input)
