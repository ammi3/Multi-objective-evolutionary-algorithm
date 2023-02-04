import random, copy, math, turtle, os
from Tool import myRandom, myFileOperator
import matplotlib.pyplot as plt
import numpy as np

class MOEAD:
    def __init__(self, popSize, maxGen, T, taskNumberRange):
        self.popSize = popSize
        self.maxGen = maxGen
        self.T = T  # 邻居数量
        self.taskNumberRange = taskNumberRange
        self.VT = {}  # 权重向量集合
        self.B = {}  # 权向量的邻居
        self.population = []
        self.Z = []  # 参考点
        self.objectNumber = 2
        self.F_rank = []  # 将种群非支配排序分层, 用种群中的个体的下标来表示，一个元素表示第一层,下标从1开始

        self.PF_history = []            #每一代的历 史最优Pareto Front
        self.EP = []                    #保存当前代的历史非支配解
        #---------------------------------------Problem Notation----------------------------------------
        self.f_MEC = 4  #Computation capacity of [10,20] server
        self.MEC_radius = 100
        self.SeNB_radius = 50
        self.SeNBSet = []

        self.Bandwidth = 20             #Bandwidth
        self.N = 10             #Number of channel
        self.w = (self.Bandwidth / self.N) * pow(10, 6)  #The bandwidth of channel
        self.noisePower = pow(10, -176/10)*pow(10, -3)  #The background noise power (50dBm = 100w)
        self.kk = pow(10, -11)  #It is a coefficient depending on the chip architecture
        self.totalSMDNumber = 0  # The total number of SMD in the network system
        self.codeLength = 0

        self.H = 3  # The number of the core in a SMD.
        self.readMECNetwork()
        self.M = self.SeNBSet.__len__()
        self.calculateInterference()
        self.calculateDataTransmissionRate()
        print("The total SMD number: ", self.totalSMDNumber)
        print("The code length: ", self.codeLength)




    def run(self):
        self.initializeWeightVectorAndNeighbor()
        self.initializePopulation()
        self.initializeReferencePoint()
        self.fast_non_dominated_sort(self.population)
        self.initializeEP(self.F_rank[1])

        t = 1
        while (t <= self.maxGen):
            print('Generation--', t)

            for i in range(self.popSize):
                y_ = self.reproduction(i)
                self.updateReferencePoint(y_)
                self.updateNeighborSolutions(i, y_)
                self.update_EP_FromElement(self.EP, y_)
            t += 1



        for ep in self.EP:
            ep.temp_fitness = ep.fitness[0]
        test_fast = sorted(self.EP, key=lambda Pareto: Pareto.temp_fitness)
        EP_list = [ind.fitness for ind in test_fast]
        return EP_list  # 返回最终的非支配解集

    """
        **********************************************run**********************************************
    """


    def initializeEP(self, F_rank):
        for ind in F_rank:
            pareto = Pareto()
            pareto.fitness = copy.deepcopy(ind.fitness)
            self.EP.append(pareto)


    def initializeWeightVectorAndNeighbor(self):
        H = self.popSize - 1
        for i in range(0, H+1):
            w = []
            w1 = i/H - 0.0
            w2 = 1.0 - i/H
            w.append(w1)
            w.append(w2)
            self.VT[i] = w

        for i in self.VT.keys():
            distance = []
            for j in self.VT.keys():
                if(i != j):
                    tup = (j, self.getDistance(self.VT[i], self.VT[j]))
                    distance.append(tup)
            distance= sorted(distance, key=lambda x:x[1])
            neighbor = []
            for j in range(self.T):
                neighbor.append(distance[j][0])
            self.B[i] = neighbor



    def initializePopulation(self):
        for i in range(int(self.popSize/2)):  # Generate randomly
            ind = Individual()
            for senb in self.SeNBSet:
                for smd in senb.SMDSet:
                    temp_smd = copy.deepcopy(smd)
                    for j in range(temp_smd.workflow.taskNumber):
                        temp_smd.workflow.position.append(random.randint(1, self.H+1))
                    temp_smd.workflow.sequence = self.initializeWorkflowSequence(temp_smd.workflow)
                    ind.chromosome.append(temp_smd)
            self.calculateFitness(ind)
            self.population.append(ind)

        for i in range(int(self.popSize/2)):
            ind = Individual()
            for senb in self.SeNBSet:
                for smd in senb.SMDSet:
                    temp_smd = copy.deepcopy(smd)
                    for j in range(temp_smd.workflow.taskNumber):
                        T_local = (temp_smd.workflow.taskSet[j].c_i_j_k/temp_smd.coreCC[1] + \
                                  temp_smd.workflow.taskSet[j].c_i_j_k/temp_smd.coreCC[2]+\
                                  temp_smd.workflow.taskSet[j].c_i_j_k/temp_smd.coreCC[3])/3
                        T_cloud = (temp_smd.workflow.taskSet[j].d_i_j_k+temp_smd.workflow.taskSet[j].o_i_j_k)/temp_smd.R_i_j \
                                  + temp_smd.workflow.taskSet[j].c_i_j_k/temp_smd.coreCC[4]
                        if T_cloud<=T_local:
                            temp_smd.workflow.position.append(self.H + 1)
                        else:
                            temp_smd.workflow.position.append(random.randint(1, self.H))
                    temp_smd.workflow.sequence = self.initializeWorkflowSequence(temp_smd.workflow)
                    ind.chromosome.append(temp_smd)
            self.calculateFitness(ind)
            self.population.append(ind)


    def initializeReferencePoint(self):
        fitness_1 = [] #存储所有个体的第一个适应度值
        fitness_2 = [] #存储所有个体的第二个适应度值
        for ind in self.population:
            fitness_1.append(ind.fitness[0])
            fitness_2.append(ind.fitness[1])
        self.Z.append(min(fitness_1))
        self.Z.append(min(fitness_2))


    def reproduction(self, i):
        k = random.choice(self.B[i])
        l = random.choice(self.B[i])
        ind_k = Individual()
        ind_l = Individual()
        for gene in self.population[k].chromosome:
            smd_k = copy.deepcopy(gene)
            self.reInitialize_WorkflowTaskSet_Schedule(smd_k)
            ind_k.chromosome.append(smd_k)

        for gene in self.population[l].chromosome:
            smd_l = SMD()
            smd_l.workflow.position = copy.copy(gene.workflow.position)
            smd_l.workflow.sequence = copy.copy(gene.workflow.sequence)
            ind_l.chromosome.append(smd_l)

        self.crossoverOperator(ind_k, ind_l)
        self.mutantOperator(ind_k)
        self.calculateFitness(ind_k)
        return ind_k


    def crossoverOperator(self, ind_k, ind_l):
        for i in range(self.totalSMDNumber):
            gene_1 = ind_k.chromosome[i]
            gene_2 = ind_l.chromosome[i]
            cpt = random.randint(0, len(gene_1.workflow.position) - 1)
            cPart_1 = []  # 保存第一个个体的执行顺序的从开始到交叉点的片段
            cPart_2 = []  # 保存第二个个体的执行顺序的从开始到交叉点的片段
            # 执行位置交叉
            for j in range(0, cpt):
                gene_1.workflow.position[j], gene_2.workflow.position[j] = gene_2.workflow.position[j], gene_1.workflow.position[j]
                cPart_1.append(gene_1.workflow.sequence[j])
                cPart_2.append(gene_2.workflow.sequence[j])
            # 执行顺序交叉
            for j in range(len(cPart_1)):
                gene_2.workflow.sequence.remove(cPart_1[j])  # 在个体二中移除第一个个体的交叉片段
                gene_1.workflow.sequence.remove(cPart_2[j])  # 在个体一中移除第二个个体的交叉片段
            gene_1.workflow.sequence = cPart_2 + gene_1.workflow.sequence
            gene_2.workflow.sequence = cPart_1 + gene_2.workflow.sequence


    def mutantOperator(self, ind):
        for gene in ind.chromosome:
            rnd_SMD = myRandom.get_0to1_RandomNumber()
            if (rnd_SMD < 1.0 / self.totalSMDNumber):  # 针对每一个基因（SMD）判断是否变异
                for i in range(len(gene.workflow.position)):
                    rnd_bit = myRandom.get_0to1_RandomNumber()
                    if (rnd_bit < 1.0 / (gene.workflow.taskNumber)):
                        pos = gene.workflow.position[i]
                        rand = [1, 2, 3, 4]
                        rand.remove(pos)
                        gene.workflow.position[i] = random.choice(rand)

                wrap_list = [] # save two adjacent task id without precedence relationship
                for j in range(1, len(gene.workflow.sequence)-1):
                    first_taskId = gene.workflow.sequence[j]
                    second_taskId = gene.workflow.sequence[j+1]
                    if first_taskId not in gene.workflow.taskSet[second_taskId].preTaskSet:
                        wrap_list.append((j, j+1))
                if wrap_list != []:
                    index = np.random.randint(0, len(wrap_list))
                    first_taskIndex = wrap_list[index][0]
                    second_taskIndex = wrap_list[index][1]
                    gene.workflow.sequence[first_taskIndex], gene.workflow.sequence[second_taskIndex] =\
                        gene.workflow.sequence[second_taskIndex], gene.workflow.sequence[first_taskIndex]
                else:
                    gene.workflow.sequence = self.initializeWorkflowSequence(gene.workflow)


    def updateReferencePoint(self, y_):
        for j in range(self.objectNumber):
            if(self.Z[j] > y_.fitness[j]):
                self.Z[j] = y_.fitness[j]


    def updateNeighborSolutions(self, i, y_):
        for j in self.B[i]:
            y_g_te = self.getTchebycheffValue(j, y_)
            neig_g_te = self.getTchebycheffValue(j, self.population[j])
            if(y_g_te <= neig_g_te):
                self.population[j] = y_


    def update_EP_FromElement(self, EP, ind):  #用新解ind来更新EP
        if EP == []:
            pareto = Pareto()
            pareto.fitness = copy.deepcopy(ind.fitness)
            EP.append(pareto)
        else:
            i = 0
            while (i < len(EP)):  # 判断ind是否支配EP中的非支配解，若支配，则删除它所支配的解
                if (self.isDominated(ind.fitness, EP[i].fitness) == True):
                    EP.remove(EP[i])
                    i -= 1
                i += 1
            for ep in EP:
                if (self.isDominated(ep.fitness, ind.fitness) == True):
                    return None
            if (self.isExist(ind, EP) == False):
                pareto = Pareto()
                pareto.fitness = copy.deepcopy(ind.fitness)
                EP.append(pareto)

    def getTchebycheffValue(self, index, ind):  #index是fitness个体的索引，用来获取权重向量
        g_te = []
        for i in range(self.objectNumber):
            temp = self.VT[index][i] * abs(ind.fitness[i] - self.Z[i])
            g_te.append(temp)
        return max(g_te)


    def printSMDTask(self, ind):
        for smd in ind.chromosome:
            print("Transmission Rate: ", smd.R_i_j)
            print("position: ", smd.workflow.position)
            print("sequence: ", smd.workflow.sequence)
            self.printWorkflowSchedule(smd.workflow)
        print("\n")


    def plotNonDominatedSolution(self, EP):
        x = []
        y = []
        for ep in EP:
            x.append(ep.fitness[0])
            y.append(ep.fitness[1])

        # plt.figure(figsize=(7, 5))
        plt.scatter(x, y, marker='o')
        plt.grid(True)
        plt.xlabel('Makespan (s)')
        plt.ylabel('Energy Consumption (w)')
        plt.title('Non-dominated Solution')
        plt.show()


    def update_EP_FromSet(self, EP, F_rank):  #用当前代的非支配排序后的第一层的非支配解来更新EP
        if(EP == []):
            for ind in F_rank:
                EP.append(ind)
        else:
            for ind in F_rank:
                if(self.isExist(ind, EP) == False):                 # 先判断ind是否在EP中，若在，则返回True。
                    if(self.isEP_Dominated_ind(ind, EP) == False):  # 然后再判断EP是否支配ind
                        i = 0
                        while(i<EP.__len__()):  #判断ind是否支配EP中的非支配解，若支配，则删除它所支配的解
                            if (self.isDominated(ind.fitness, EP[i].fitness) == True):
                                EP.remove(EP[i])
                                i -= 1
                            i += 1
                        EP.append(ind)


    def update_PF_history(self, PF_history, EP):  # Updating PF_history using EP
        PF = []
        for ep in EP:
            PF.append(list(ep.fitness))
        self.PF_history.append(PF)


    def isExist(self, ind, EP):   #判断个体ind的适应度是否与EP中某个个体的适应度相对，若相等，则返回True
        for ep in EP:
            if ind.fitness == ep.fitness: # 判断两个列表对应元素的值是否相等
                return True
        return False


    def isEP_Dominated_ind(self, ind, EP):   #判断EP中的某个个体是否支配ind，若支配，则返回True
        for ep in EP:
            if self.isDominated(ep.fitness, ind.fitness):
                return True
        return False


    def fast_non_dominated_sort(self, population):
        for p in population:
            p.S_p = []
            p.rank = None
            p.n = 0

        self.F_rank = []
        F1 = []  # 第一个非支配解集前端
        self.F_rank.append(None)
        for p in population:
            for q in population:
                if self.isDominated(p.fitness, q.fitness):
                    p.S_p.append(q)
                elif self.isDominated(q.fitness, p.fitness):
                    p.n += 1
            if (p.n == 0):
                p.rank = 1
                F1.append(p)
        self.F_rank.append(F1)

        i = 1
        while (self.F_rank[i] != []):
            Q = []
            for p in self.F_rank[i]:
                for q in p.S_p:
                    q.n -= 1
                    if (q.n == 0):
                        q.rank = i + 1
                        Q.append(q)

            if(Q != []):
                i += 1
                self.F_rank.append(Q)
            else:
                break


    def isDominated(self, fitness_1, fitness_2):  # 前者是否支配后者
        flag = -1
        for i in range(self.objectNumber):
            if fitness_1[i] < fitness_2[i]:
                flag = 0
            if fitness_1[i] > fitness_2[i]:
                return False
        if flag == 0:
            return True
        else:
            return False





    def calculatePopulationFitness(self, population):
        for ind in population:
            self.calculateFitness(ind)


    def calculateFitness(self, ind):
        ind.fitness = []
        time = []
        energy = []
        for gene in ind.chromosome:
            smd = gene  #一个gene就是一个SMD
            self.calculateWorkflowTimeEnergy(smd, smd.workflow)
            time.append(smd.workflow.schedule.T_total)
            energy.append(smd.workflow.schedule.E_total)
        ind.fitness.append(np.average(time))
        ind.fitness.append(np.average(energy))


    def calculateWorkflowTimeEnergy(self, smd, workflow):
        workflow.schedule.TimeEnergy = []
        workflow.schedule.T_total = None
        workflow.schedule.E_total = 0

        for i in range(len(workflow.sequence)):
            taskId = workflow.sequence[i]
            pos = workflow.position[i]
            task = workflow.taskSet[taskId]
            task.exePosition = pos
            if pos == 4:   # The task is executed on [10,20] server.
                task.islocal = False
                if task.id == workflow.entryTask:
                    task.RT_i_l = task.ST_i_l = task.FT_i_l = 0
                    task.RT_i_ws = task.ST_i_ws = 0.0
                    task.FT_i_ws = task.ST_i_ws + task.d_i_j_k/smd.R_i_j
                    task.RT_i_c = task.ST_i_c = task.FT_i_ws
                    task.FT_i_c = task.ST_i_c + task.c_i_j_k/smd.coreCC[pos]
                    task.RT_i_wr = task.ST_i_wr = task.FT_i_c
                    task.FT_i_wr = task.ST_i_wr + task.o_i_j_k/smd.R_i_j
                    workflow.schedule.wsTP.append(task.FT_i_ws)
                    workflow.schedule.MECTP.append(task.FT_i_c)
                    workflow.schedule.wrTP.append(task.FT_i_wr)
                else:
                    task.RT_i_ws = self.get_RT_i_ws(task, workflow)
                    task.ST_i_l = float("inf")
                    task.FT_i_l = float("inf")
                    if workflow.schedule.wsTP[-1] < task.RT_i_ws:
                        task.ST_i_ws = task.RT_i_ws
                        task.FT_i_ws = task.ST_i_ws + task.d_i_j_k/smd.R_i_j
                    else:
                        task.ST_i_ws = workflow.schedule.wsTP[-1]
                        task.FT_i_ws = task.ST_i_ws + task.d_i_j_k/smd.R_i_j
                    workflow.schedule.wsTP.append(task.FT_i_ws)

                    task.RT_i_c = self.get_RT_i_c(task, workflow)
                    if workflow.schedule.MECTP[-1] < task.RT_i_c:
                        task.ST_i_c = task.RT_i_c
                        task.FT_i_c = task.ST_i_c + task.c_i_j_k/smd.coreCC[pos]
                    else:
                        task.ST_i_c = workflow.schedule.MECTP[-1]
                        task.FT_i_c = task.ST_i_c + task.c_i_j_k/smd.coreCC[pos]
                    workflow.schedule.MECTP.append(task.FT_i_c)

                    task.RT_i_wr = task.FT_i_c
                    if workflow.schedule.wrTP[-1] < task.RT_i_wr:
                        task.ST_i_wr = task.RT_i_wr
                        task.FT_i_wr = task.ST_i_wr + task.o_i_j_k/smd.R_i_j
                    else:
                        task.ST_i_wr = workflow.schedule.wrTP[-1]
                        task.FT_i_wr = task.ST_i_wr + task.o_i_j_k/smd.R_i_j
                    workflow.schedule.wrTP.append(task.FT_i_wr)
                task.energy += smd.pws_i_j * (task.FT_i_ws - task.ST_i_ws)
                task.energy += smd.pwr_i_j * (task.FT_i_wr - task.ST_i_wr)
                workflow.schedule.E_total += task.energy
            else:          # The task is executed on a local core.
                task.islocal = True
                task.RT_i_ws = task.RT_i_c = task.RT_i_wr = 0.0
                task.ST_i_ws = task.ST_i_c = task.ST_i_wr = 0.0
                task.FT_i_ws = task.FT_i_c = task.FT_i_wr = 0.0
                if task.id == workflow.entryTask:
                    task.RT_i_l = task.ST_i_l = 0
                    task.FT_i_l = task.ST_i_l + task.c_i_j_k/smd.coreCC[pos]
                else:
                    task.RT_i_l = self.get_RT_i_l(task, workflow)
                    if task.RT_i_l > workflow.schedule.coreTP[pos][-1]:
                        task.ST_i_l = task.RT_i_l
                    else:
                        task.ST_i_l = workflow.schedule.coreTP[pos][-1]
                    task.FT_i_l = task.ST_i_l + task.c_i_j_k/smd.coreCC[pos]
                workflow.schedule.coreTP[pos].append(task.FT_i_l)
                task.energy = smd.pcc_i_j[pos] * (task.FT_i_l - task.ST_i_l)
                workflow.schedule.E_total += task.energy
            workflow.schedule.S[pos].append(task.id)
        if workflow.taskSet[workflow.exitTask].islocal == True:
            workflow.schedule.T_total = workflow.taskSet[workflow.exitTask].FT_i_l
        else:
            workflow.schedule.T_total = workflow.taskSet[workflow.exitTask].FT_i_wr
        workflow.schedule.TimeEnergy.append(workflow.schedule.T_total)
        workflow.schedule.TimeEnergy.append(workflow.schedule.E_total)


    def get_RT_i_ws(self, task, workflow):
        if task.id == workflow.entryTask:
            return 0.0
        else:
            pre_max = []
            for pre_taskId in task.preTaskSet:
                if workflow.taskSet[pre_taskId].islocal == True:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_l)
                else:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_ws)
            return max(pre_max)


    def get_RT_i_c(self, task, workflow):
        pre_max = []
        for pre_taskId in task.preTaskSet:
            pre_max.append(workflow.taskSet[pre_taskId].FT_i_c)
        return max(task.FT_i_ws, max(pre_max))


    def get_RT_i_l(self, task, workflow):
        if task.id == workflow.entryTask:
            return 0.0
        else:
            pre_max = []
            for pre_taskId in task.preTaskSet:
                if workflow.taskSet[pre_taskId].islocal == True:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_l)
                else:
                    pre_max.append(workflow.taskSet[pre_taskId].FT_i_wr)
            return max(pre_max)


    def reInitialize_WorkflowTaskSet_Schedule(self, smd):
        for task in smd.workflow.taskSet:
            self.reInitializeTaskSet(task)
        self.reInitializeSchedule(smd.workflow.schedule)


    def reInitializeTaskSet(self, task):
        task.islocal = None
        task.exePosition = None
        task.RT_i_l = task.ST_i_l = task.FT_i_l = None
        task.RT_i_ws = task.RT_i_c = task.RT_i_wr = None
        task.ST_i_ws = task.ST_i_c = task.ST_i_wr = None
        task.FT_i_ws = task.FT_i_c = task.FT_i_wr = None
        task.energy = 0


    def reInitializeSchedule(self, schedule):
        schedule.S = {1:[], 2:[], 3:[], 4:[]}
        schedule.coreTP = {1:[0], 2:[0], 3:[0]}
        schedule.wsTP = [0]
        schedule.MECTP = [0]
        schedule.wrTP = [0]
        schedule.T_total = None
        schedule.E_total = 0
        schedule.TimeEnergy = []


    def initializeWorkflowSequence(self, workflow):
        S = []  # 待排序的任务集合
        R = []  # 已排序任务
        T = []
        R.append(workflow.entryTask)
        for task in workflow.taskSet:
            T.append(task.id)
        T.remove(workflow.entryTask)

        while T != []:
            for t in T:
                if set(workflow.taskSet[t].preTaskSet).issubset(set(R)):  #判断t的前驱节点集是否包含在R中
                    if t not in S:
                        S.append(t)
            ti = random.choice(S) #随机从S中选择一个元素
            S.remove(ti)
            T.remove(ti)
            R.append(ti)
        return R


    def printWorkflowSchedule(self, workflow):
        print("S--", workflow.schedule.S)
        for coreId in workflow.schedule.S:
            if coreId < 4:
                print("core", coreId, ": ",end="")
                for taskId in workflow.schedule.S[coreId]:
                    task = workflow.taskSet[taskId]
                    print(str(taskId)+"=("+str(round(task.ST_i_l,2))+","+str(round(task.FT_i_l,2))+") ", end="")
                print("\n")

        for coreId in workflow.schedule.S:
            if coreId == 4:
                print("WS:      ", end="")
                for taskId in workflow.schedule.S[coreId]:
                    task = workflow.taskSet[taskId]
                    print(str(taskId)+"=("+str(round(task.ST_i_ws,2))+","+str(round(task.FT_i_ws,2))+") ", end="")
                break
        print("\n")

        for coreId in workflow.schedule.S:
            if coreId == 4:
                print("Cloud:   ", end="")
                for taskId in workflow.schedule.S[coreId]:
                    task = workflow.taskSet[taskId]
                    print(str(taskId) + "=(" + str(round(task.ST_i_c,2)) + "," + str(round(task.FT_i_c,2)) + ") ",end="")
                break
        print("\n")

        for coreId in workflow.schedule.S:
            if coreId == 4:
                print("WR:      ", end="")
                for taskId in workflow.schedule.S[coreId]:
                    task = workflow.taskSet[taskId]
                    print(str(taskId) + "=(" + str(round(task.ST_i_wr,2)) + "," + str(round(task.FT_i_wr,2)) + ") ",end="")
                break
        print("\n")
        print("(Time, Energy)=", workflow.schedule.TimeEnergy)
        print("\n\n")


    def calculateInterference(self):
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
               I_i_j = 0
               for m in range(self.M):
                   if(self.SeNBSet[m] != self.SeNBSet[i]):
                       for k in range(self.SeNBSet[m].SMDNumber):
                           if(self.SeNBSet[m].SMDSet[k].channel == self.SeNBSet[i].SMDSet[j].channel):  # U_m_j and U_l_i have the same channel
                               g_i_m_k = self.getChannelGain(self.SeNBSet[m].SMDSet[k].coordinate, self.SeNBSet[i].coordinate)
                               I_i_j += self.SeNBSet[m].SMDSet[k].pws_i_j * g_i_m_k
               self.SeNBSet[i].SMDSet[j].I_i_j = I_i_j


    def calculateDataTransmissionRate(self):
        self.calculateChannelGain()
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
                log_v = 1 + (self.SeNBSet[i].SMDSet[j].pws_i_j*self.SeNBSet[i].SMDSet[j].g_i_j) / (self.noisePower + self.SeNBSet[i].SMDSet[j].I_i_j)
                self.SeNBSet[i].SMDSet[j].R_i_j = self.w * math.log(log_v, 2)


    def calculateChannelGain(self):  #calculate G_m_j between SMD U_m_j and SeNB S_m
        for i in range(self.M):
            for j in range(self.SeNBSet[i].SMDNumber):
                self.SeNBSet[i].SMDSet[j].g_i_j = self.getChannelGain(self.SeNBSet[i].SMDSet[j].coordinate, self.SeNBSet[i].coordinate)


    def getChannelGain(self, U_i_j, S_i):  # channel gain= D^(-pl), where D is the distance between U_m_j and S_m, pl=4 is the path loss factor
        distance = self.getDistance(U_i_j, S_i)
        channelGain = pow(distance, -4)
        return channelGain


    def getDistance(self, point1, point2):
        return np.sqrt(np.sum(np.square([point1[i] - point2[i] for i in range(2)])))


    def getWorkflow(self, filename):
        wf = Workflow()
        with open(filename, 'r') as readFile:
            for line in readFile:
                task = Task()
                s = line.splitlines()
                s = s[0].split(':')
                predecessor = s[0]
                id = s[1]
                successor = s[2]
                if (predecessor != ''):
                    predecessor = predecessor.split(',')
                    for pt in predecessor:
                        task.preTaskSet.append(int(pt))
                else:
                    wf.entryTask = int(id)
                task.id = int(id)
                if (successor != ''):
                    successor = successor.split(',')
                    for st in successor:
                        task.sucTaskSet.append(int(st))
                else:
                    wf.exitTask = int(id)
                wf.taskSet.append(task)
        return wf


    def readMECNetwork(self):
        file_SMD_task_cpu = open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\SMD_Task_CPU_Cycles_Number.txt', 'r')
        file_SMD_task_data = open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\SMD_Task_Data_Size.txt', 'r')
        file_SMD_output_task_data = open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\SMD_Task_Output_Data_Size.txt', 'r')

        SeNB_count = -1
        with open(self.getCurrentPath() + '\\' + self.taskNumberRange + '\MEC_Network.txt', 'r') as readFile:
            for line in readFile:
                if(line == '---file end---\n'):
                    break
                elif(line == 'SeNB:\n'):
                    SeNB_count += 1
                    senb = SeNB()     #create SeNB cell
                    if(readFile.readline() == 'Coordinate:\n'):
                        SeNB_crd = readFile.readline()
                        SeNB_crd = SeNB_crd.splitlines()
                        SeNB_crd = SeNB_crd[0].split('  ')
                        senb.coordinate.append(float(SeNB_crd[0]))
                        senb.coordinate.append(float(SeNB_crd[1]))

                        if(readFile.readline() == 'SMD number:\n'):
                            senb.SMDNumber = int(readFile.readline())

                        for line1 in readFile:
                            if (line1 == '---SeNB end---\n'):
                                break
                            elif(line1 == 'SMD:\n'):
                                self.totalSMDNumber += 1
                                smd = SMD()
                                if (readFile.readline() == 'Coordinate:\n'):
                                    SMD_crd = readFile.readline()
                                    SMD_crd = SMD_crd.splitlines()
                                    SMD_crd = SMD_crd[0].split('  ')
                                    smd.coordinate.append(float(SMD_crd[0]))
                                    smd.coordinate.append(float(SMD_crd[1]))

                                if (readFile.readline() == 'Computation capacity:\n'):
                                    SMD_cc = readFile.readline()
                                    SMD_cc = SMD_cc.splitlines()
                                    SMD_cc = SMD_cc[0].split('  ')
                                    smd.coreCC[1] = float(SMD_cc[0])
                                    smd.coreCC[2] = float(SMD_cc[1])
                                    smd.coreCC[3] = float(SMD_cc[2])

                                if (readFile.readline() == 'The number of task:\n'):  #在SeNB（SeNB_count）下得到一个工作流
                                    taskNumber = int(readFile.readline())
                                    SeNB_directory = "SeNB-"+str(SeNB_count)+"\\t"+str(taskNumber)+".txt"
                                    wf_directory = self.getCurrentPath()+"\workflowSet\\"+SeNB_directory
                                    smd.workflow = self.getWorkflow(wf_directory)
                                    smd.workflow.taskNumber = taskNumber
                                    self.codeLength += taskNumber
                                    for task in smd.workflow.taskSet:
                                        task.c_i_j_k = float(file_SMD_task_cpu.readline())    #读取执行任务需要的cpu循环数量
                                        task.d_i_j_k = float(file_SMD_task_data.readline()) * 1024
                                        task.o_i_j_k = float(file_SMD_output_task_data.readline()) * 1024

                                if (readFile.readline() == 'Channel:\n'):
                                    channel = readFile.readline()
                                    smd.channel = int(channel)

                                senb.SMDSet.append(smd)
                    self.SeNBSet.append(senb)
        file_SMD_task_data.close()
        file_SMD_task_cpu.close()


    def getMECNetworkSMDNumber(self):
        sum_SMDNumber = 0
        for senb in self.SeNBSet:
            sum_SMDNumber += senb.SMDNumber
        return (sum_SMDNumber)


    def plotMECNetwork(self):
        turtle.setup(width=500, height=500, startx=10, starty=10)
        turtle.pencolor('red')
        turtle.dot(18)
        turtle.penup()
        turtle.goto(0, -self.MEC_radius)
        turtle.pendown()
        turtle.pensize(3)
        turtle.pencolor('black')
        turtle.speed(5)
        turtle.circle(self.MEC_radius)
        for senb in self.SeNBSet:
            turtle.pencolor('blue')
            turtle.penup()
            turtle.goto(senb.coordinate[0], senb.coordinate[1])
            turtle.pendown()
            turtle.dot(10)

            turtle.penup()
            turtle.goto(senb.coordinate[0], senb.coordinate[1]-self.SeNB_radius)
            turtle.pendown()
            turtle.circle(50)
            turtle.pencolor('green')
            for smd in senb.SMDSet:
                turtle.penup()
                turtle.goto(smd.coordinate[0], smd.coordinate[1])
                turtle.pendown()
                turtle.dot(10)
        # 点击窗口关闭
        window = turtle.Screen()
        window.exitonclick()


    def printPopulationFitness(self, population):
        for ind in population:
            print('Index:  ', population.index(ind), "--",  ind.fitness)

    def getCurrentPath(self):
        return os.path.dirname(os.path.realpath(__file__))

    def getProjectPath(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(os.path.dirname(cur_path))




class Individual:
    def __init__(self):
        self.chromosome = []      #基因位是SMD类型
        self.fitness = []
        self.isFeasible = True    #判断该个体是否合法
        self.temp_fitness = None  #临时适应度，计算拥挤距离的时候，按每个目标值来对类列表进行升序排序
        self.distance = 0.0
        self.rank = None
        self.S_p = []  #种群中此个体支配的个体集合
        self.n = 0  #种群中支配此个体的个数

class SeNB:
    def __init__(self):
        self.coordinate = []  #The position coordination of the SeNB
        self.SMDNumber = 0
        self.SMDSet = []    #The set of SMD the SeNB covers

class SMD:
    def __init__(self):
        self.coordinate = []    # The position coordination of the SMD
        self.workflow = Workflow()      #The workflow of the SMD
        self.channel = None     # Gaining channel index
        self.g_i_j = None       # The channel gain between the SMD and SeNB Sm
        self.R_i_j = None       # The data transmission rate of the SMD
        self.I_i_j = None       # The interference at the SMD
        #The SMD is modeled as a 3-tuple
        self.coreCC = {1:None, 2:None, 3:None, 4:4}        # The computing capacity of three core.

        self.pcc_i_j = {1:4, 2:2, 3:1}  # The power consumption of the three cores under the maximum operating frequency.
        self.pws_i_j = 0.5  # The send data power (w) of the SMD
        self.pwr_i_j = 0.1  # The receive data power (w) of the SMD

class Workflow:
    def __init__(self):
        self.entryTask = None      #开始任务
        self.exitTask = None       #结束任务
        self.position = []         #执行位置
        self.sequence = []         #执行顺序
        self.taskNumber = None
        self.taskSet = []          #列表的索引值就是任务的id值
        self.schedule = Schedule()

class Schedule:
    def __init__(self):
        self.taskSet = {}
        self.S = {1:[], 2:[], 3:[], 4:[]} # Record the set of task that is executed certain execution unit selection. eg. S[3]=[v1,v3,v5,v7,v9,v10]
        self.coreTP = {1:[0], 2:[0], 3:[0]}  # Index is core number, its element denotes the current time point on the core.
        self.wsTP = [0]  # The current time point on the wireless sending channel.
        self.MECTP = [0]  # The current time point on the cloud.
        self.wrTP = [0]  # The current time point on the wireless receiving channel.
        self.T_total = None
        self.E_total = 0
        self.TimeEnergy = []

class Task:
    def __init__(self):
        self.id = None
        self.islocal = None    # Denote the task is executed locally or on cloud.
        self.preTaskSet = []   #The set of predecessor task (element is Task class).
        self.sucTaskSet = []   #The set of successor task (element is Task class).
        self.exePosition = None  # it denotes execution position (i.e., [1,2,3,4])of the task.
        self.actualFre = 1    # The actual frequency scaling factors.
        self.c_i_j_k = None    # The number of CPU cycles required to perform task
        self.d_i_j_k = None    # The data size of the task.
        self.o_i_j_k = None    # The output data size of the task.

        self.RT_i_l = None     # The ready time of task vi on a local core.
        self.RT_i_ws = None    # The ready time of task vi on the wireless sending channel.
        self.RT_i_c = None     # The ready time of task vi on the [10,20] server.
        self.RT_i_wr = None    # The ready time for the cloud to transmit back the results of task vi

        self.ST_i_l = None     # The start time of task vi on a local core.
        self.ST_i_ws = None    # The start time of task vi on the wireless sending channel.
        self.ST_i_c = None     # The start time of task vi on the [10,20] server.
        self.ST_i_wr = None    # The start time for the cloud to transmit back the results of task vi

        self.FT_i_l = None     # The finish time of task vj on a local core.
        self.FT_i_ws = None    # The finish time of task vj on the wireless sending channel.
        self.FT_i_c = None     # The finish time of task vj on the [10,20] server.
        self.FT_i_wr = None    # The finish time of task vj on the wireless receiving channel.
        self.energy = 0


class Pareto:
    def __init__(self):
        self.chromosome = None
        self.fitness = []
        self.temp_fitness = None  #排序使用

