import matplotlib.pyplot as plt
import math
import random

"""
函数里面所有以plot开头的函数都可以注释掉，没有影响
求解的目标表达式为：
y = x**2
"""


def main():
    pop_size = 1000                                                                                          # 种群规模，种群数量如果是个位数或2位数，容易产生适应度为0的情况。
    upper_limit = 4                                                                                         # 基因中允许出现的最大值
    chromosome_length = 6                                                                                   # 染色体长度为6
    iter = 100                                                                                              # 迭代次数，即需要进化多少代
    pc = 0.7                                                                                                # 杂交概率
    pm = 0.17                                                                                               # 变异概率，一般为编码长度的倒数
    results = []                                                                                            # 存储每一代的最优解，N个二元组
    best_X = []
    best_Y = []
    pop = init_population(pop_size, chromosome_length)                                                      # 初始化种群
    for i in range(iter):
        obj_value = calc_obj_value(pop, chromosome_length, upper_limit)                                     # 个体评价，有负值
        fit_value = calc_fit_value(obj_value)                                                               # 个体适应度，不好的归0，可以理解为去掉上面的负值
        best_individual, best_fit = find_best(pop, fit_value)                                               # 第一个是最优基因序列, 第二个是对应的最佳个体适度
        results.append([binary2decimal(best_individual, upper_limit, chromosome_length), best_fit])
        best_X.append(results[-1][0])
        best_Y.append(results[-1][1])
        selection(pop, fit_value)                                                                           # 选择
        crossover(pop, pc)                                                                                  # 染色体交叉（最优个体之间进行0、1互换）
        mutation(pop, pm)                                                                                   # 染色体变异（其实就是随机进行0、1取反）
    else:
	    print(results)                                                                               # 打印出每一代的最优值
	    plot_best_by_generation(best_X, best_Y)

def plot_best_by_generation(X, Y):                                                                          # 绘制每一代的最优值
    plt.plot(X,Y)
    plt.show()

def binary2decimal(binary, upper_limit, chromosome_length):                                                 # 用于解码中的2进制转十进制运算
    t = 0
    for j in range(len(binary)):
        t += binary[j] * 2 ** j
    t = t * upper_limit / (2 ** chromosome_length - 1)
    return t

def encode_chromosome(pop_size, chromosome_length):
		pop_encode = [[random.randint(0, 1) for i in range(chromosome_length)] for j in range(pop_size)]    #按照种群数量生成对应的每个个体的二进制编码
		return pop_encode

def init_population(pop_size, chromosome_length):
    pop = encode_chromosome(pop_size, chromosome_length)
    return pop


def decode_chromosome(pop, chromosome_length, upper_limit):                                                 # 解码
    X = []
    for ele in pop:
        temp = 0
        for i, coff in enumerate(ele):
            temp += coff * (2 ** i)
        X.append(temp * upper_limit / (2 ** chromosome_length - 1))
    return X


def calc_obj_value(pop, chromosome_length, upper_limit):                                                    # 计算目标函数值
    obj_value = []
    X = decode_chromosome(pop, chromosome_length, upper_limit)
    for x in X:
        obj_value.append(2*x**2)
    return obj_value


def calc_fit_value(obj_value):  # 物竞：使用目标函数值作为适应度
    fit_value = []
    # 去掉小于0的值，更改c_min会改变淘汰的下限
    # 比如设成10可以加快收敛
    # 但是如果设置过大，有可能影响了全局最优的搜索
    c_min = 2
    for value in obj_value:
        if value > c_min:
            fit_value.append(value)
        else:
	        fit_value.append(0.)
    # fit_value保存的是活下来的值
    return fit_value



def find_best(pop, fit_value):  # 找出最优解和最优解的基因编码
    # 用来存最优基因编码
    best_individual = []
    #先假设第一个基因的适应度最好
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return best_individual, best_fit

# 计算累积适应度
def cum_sum(fit_value):
    # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]，matlab的一个函数
    # 这个地方遇坑，局部变量如果赋值给引用变量，在函数周期结束后，引用变量也将失去这个值
    temp = fit_value[:]
    for i in range(len(temp)):
        fit_value[i] = (sum(temp[:i + 1]))


# 轮赌法选择
def selection(pop, fit_value):
    # p_fit_value用于存放各累计适应度的累计概率
    p_fit_value = []
    # 累计适应度总和
    total_fit = sum(fit_value)
    # 计算各适应度对应的概率
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit)

    # 计算累计适应度概率
    cum_sum(p_fit_value)
    pop_len = len(pop)
    # 类似搞一个转盘吧下面这个的意思
    ms = sorted([random.random() for i in range(pop_len)])
    fitin = 0
    newin = 0
    newpop = pop[:]
    # 转轮盘选择法
    while newin < pop_len:
        # 如果这个概率大于随机出来的那个概率，就选这个
        if (ms[newin] < p_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    # 这里注意一下，因为random.random()不会大于1，所以保证这里的newpop规格会和以前的一样
    # 而且这个pop里面会有不少重复的个体，保证种群数量一样

    # 之前是看另一个人的程序，感觉他这里有点bug，要适当修改
    pop = newpop[:]


# 杂交
def crossover(pop, pc):
    # 一定概率杂交，主要是杂交种群中相邻的两个个体
    pop_len = len(pop)
    # 由于不可能出现自己和自己杂交，故随机杂交的范围为[1,pop_len-1]
    for i in range(pop_len - 1):
        # 随机看看达到杂交概率没，此处感觉不应该是<，应该是>=，因为是达到杂交概率才做杂交，从代码上看逻辑似乎不正确
        if (random.random() < pc):
            # 随机选取杂交点，然后交换数组
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            # 取第i和第i+1个个体进行crossover操作，选取的碱基对为[0,cpoint]
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1[:]
            pop[i + 1] = temp2[:]


# 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    # 每条染色体随便选一个进行mutation操作
    for i in range(px):
        if (random.random() >= pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


if __name__ == '__main__':
    main()
