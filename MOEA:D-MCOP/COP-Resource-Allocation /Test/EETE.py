'''
W. Zhang and Y. Wen, Energy-Efficient Task Execution for Application
as a General Topology in MobileCloud Computing, IEEE Transaction on Could Computing, vol. 6, no. 3, Jul. 2018.
'''
import random, copy, math, turtle, os
from Tool import myRandom, myFileOperator
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue

class EETE:
    def __init__(self):
        # The information of workflow
        # self.DAG = {0: [1], 1: [2, 3, 4], 2: [5], 3: [6], 4: [7, 8], 5: [11], 6: [9, 10],
        # 7: [10], 8: [11], 9: [11], 10: [11], 11: [12], 12: []}
        self.DAG = {}
        self.workflow = self.getWorkflow()
        for task in self.workflow.taskSet:
            self.DAG[task.id] = task.sucTaskSet

        self.w = [0, 10, 25, 2, 12, 15, 67, 54, 24, 50, 9, 20, 0]
        self.taskNumber = len(self.workflow.taskSet)
        for i in range(self.taskNumber):
            self.workflow.taskSet[i].w_i = self.w[i]
        self.p = {(0,1):20, (1,2):30, (1,3):20, (1,4):30, (2,5):20, (3,6):50,
                  (4,7):25, (4,8):45, (5,11):60, (6,9):20, (6,10):30,
                  (7,10):10, (8,11):50, (9,11):20, (10,11):40, (11,12):10}
        self.ps = 0.1    # (W)Data transmission power of the mobile device
        self.pr = 0.05   # (W)Data receiving power of the mobile device
        self.pm = 0.5    # (W)Computation power of the mobile device
        self.p0 = 0.001  # (W)Idle power of the mobile device
        self.fm = 500    # (MHz)CPU frequency of the mobile device
        self.fc = 3000   # (MHz)CPU frequency of the mobile device
        self.R = 50      # (kb/s)Communication ratio
        self.Td = 5      # Deadline
        self.PCP = []    # Partial critical path

        self.T_total = None
        self.E_total = 0

        self.objectNumber = 2
        self.F_rank = []
        self.EP = []


    def run(self):
        self.collaborativeTaskExeSchedulingAlgorithm()




    def collaborativeTaskExeSchedulingAlgorithm(self):
        self.calculateCompAndCommCost()
        self.calculateEarliestStartTIme()
        self.calculateLatestFinishTime()
        self.setEntryAndExitTask_isScheduled()
        self.scheduleParents(self.workflow.taskSet[-1])




    def calculateCompAndCommCost(self):
        for task in self.workflow.taskSet:
            if task.id == self.workflow.entryTask:
                task.x_i = 0
                task.T_comp = task.w_i / self.fm
            elif task.id == self.workflow.exitTask:
                task.x_i = 0
                task.T_comp = task.w_i / self.fm
            else:
                task.x_i = 1
                task.T_comp = task.w_i / self.fc

        for task in self.workflow.taskSet:
            for sucTaskId in task.sucTaskSet:
                vi = task
                vj = self.workflow.taskSet[sucTaskId]
                xi = task.x_i
                xj = self.workflow.taskSet[sucTaskId].x_i
                index = (vi.id, vj.id)
                if xi == 0 and xj == 0:
                    vi.comm[index] = 0
                elif xi == 0 and xj == 1:
                    vi.comm[index] = self.p[index] / self.R
                elif xi == 1 and xj == 0:
                    vi.comm[index] = self.p[index] / self.R
                elif xi == 1 and xj == 1:
                    vi.comm[index] = 0


    def calculateEarliestStartTIme(self):
        for task in self.workflow.taskSet:
            if task.id == self.workflow.entryTask:
                task.T_es = 0
            else:
                maxTime = []
                for preTaskId in task.preTaskSet:
                    preTask = self.workflow.taskSet[preTaskId]
                    index = (preTaskId,task.id)
                    temp = preTask.T_es + preTask.T_comp + preTask.comm[index]
                    maxTime.append(temp)
                task.T_es = max(maxTime)


    def calculateLatestFinishTime(self):
        temp_DAG = copy.deepcopy(self.DAG)
        noneSuccessorNodeSet = self.getNoneSuccessorNodeSet(temp_DAG)
        while noneSuccessorNodeSet != []:
            for taskId in noneSuccessorNodeSet:
                if taskId == self.workflow.exitTask:
                    self.workflow.taskSet[taskId].T_lf = self.Td
                    temp_DAG.pop(taskId)
                else:
                    minTime = []
                    for sucTaskId in self.workflow.taskSet[taskId].sucTaskSet:
                        sucTask = self.workflow.taskSet[sucTaskId]
                        index = (taskId,sucTaskId)
                        temp = sucTask.T_lf - sucTask.T_comp - self.workflow.taskSet[taskId].comm[index]
                        minTime.append(temp)
                    self.workflow.taskSet[taskId].T_lf = min(minTime)
                    temp_DAG.pop(taskId)
            for id1 in noneSuccessorNodeSet:
                for id2 in temp_DAG:
                    if id1 in temp_DAG[id2]:
                        temp_DAG[id2].remove(id1)
            noneSuccessorNodeSet = self.getNoneSuccessorNodeSet(temp_DAG)


    def getNoneSuccessorNodeSet(self, DAG):   # Get node set without successor
        noneSuccessorNodeSet = []
        for id in DAG:
            if DAG[id] == []:
                noneSuccessorNodeSet.append(id)
        return noneSuccessorNodeSet


    def setEntryAndExitTask_isScheduled(self):
        for task in self.workflow.taskSet:
            if task.id == self.workflow.entryTask:
                task.isSchedule = True
            elif task.id == self.workflow.exitTask:
                task.isSchedule = True


    def scheduleParents(self, v):
        while self.isExistUnscheduledParent(v) == True:
            v.PCP.clear()
            v.PCP.append(v)
            u = v
            while self.isExistUnscheduledParent(u) == True:
                w = self.findCriticalParent_isUnScheduled(u)
                v.PCP.insert(0, w)
                u = w
            w_ = self.findCriticalParent_isScheduled(u)
            v.PCP.insert(0, w_)
            self.schedulePath(v.PCP)






    def isExistUnscheduledParent(self, u):
        for parentId in u.preTaskSet:
            if self.workflow.taskSet[parentId].isSchedule == False:
                return True


    def findCriticalParent_isUnScheduled(self, u):
        vi = u
        vi.criticalParent.clear()
        for vj_id in vi.preTaskSet:
            vj = self.workflow.taskSet[vj_id]
            if vj.isSchedule == False:
                index = (vj.id, vi.id)
                vj.temp_criticalParent = vj.T_es + vj.T_comp + vj.comm[index]
                vi.criticalParent.append(vj)
        vi.criticalParent = sorted(vi.criticalParent, key=lambda Task: Task.temp_criticalParent)
        return vi.criticalParent[-1]


    def findCriticalParent_isScheduled(self, u):
        for parentId in u.preTaskSet:
            parent = self.workflow.taskSet[parentId]
            if parent.isSchedule == True:
                return parent


    def schedulePath(self, PCP):
        self.printPCP(PCP)
        m = len(PCP)
        y_1 = PCP[0].x_i
        y_m = PCP[-1].x_i
        T_sd = PCP[-1].T_lf - PCP[0].T_es
        if m >= 3:
            if y_1 == 0 and y_m == 0:
                self.schedulePath_y1_0_and_ym_0(PCP, T_sd)



    def schedulePath_y1_0_and_ym_0(self, PCP, T_sd):
        PCP_case1 = copy.deepcopy(PCP)
        PCP_case2 = copy.deepcopy(PCP)
        for task in PCP_case1:
            task.x_i = 0
        energy_case1 = self.getPCP_energy(PCP_case1)

        for task in PCP_case2:
            if task.id != self.workflow.entryTask and task.id != self.workflow.exitTask:
                task.x_i = 1
        energy_case2 = self.getPCP_energy(PCP_case2)
        if energy_case1 <= energy_case2:
            for task in PCP:
                task.x_i = 0
        else:
            for task in PCP:
                if task.id != self.workflow.entryTask and task.id != self.workflow.exitTask:
                    task.x_i = 1


    def schedulePath_y1_0_and_ym_1(self, PCP):
        PCPSet = []
        for i in range(1, len(PCP)-1):
            temp_PCP = copy.deepcopy(PCP)
            for j in range(i):
                temp_PCP[j].x_i = 0
            for j in range(i, len(PCP)-1):
                temp_PCP[j].x_i = 1
            temp = []
            temp.append(temp_PCP)
            temp.append(self.getPCP_energy(temp_PCP))
            PCPSet.append(temp)



    def getPCP_energy(self, PCP):
        energy = 0
        for i in range(len(PCP)):
            task = PCP[i]
            if task.x_i == 0:
                energy += (task.w_i * self.pm) / self.fm
            else:
                energy += (task.w_i * self.p0) / self.fc
        for k in range(len(PCP) - 1):
            vi = PCP[k]
            vj = PCP[k+1]
            index = (vi.id,vj.id)
            if vi.x_i == 0 and vj.x_i == 0:
                energy += 0
            elif vi.x_i == 0 and vj.x_ == 1:
                energy += (self.ps * self.p[index]) / self.R
            elif vi.x_i == 1 and vj.x_i == 0:
                energy += (self.pr * self.p[index]) / self.R
            elif vi.x_i == 1 and vj.x_i == 0:
                energy += 0
        return energy


    def printPCP(self, PCP):
        for task in PCP:
            print(task.id, ',', end='')











#*****************************************************************************
#*****************************************************************************

    def update_EP(self, EP, F_rank):  #用当前代的非支配排序后的第一层的非支配解来更新EP
        if(EP == []):
            for ind in F_rank:
                if self.isExist(ind, EP) == False:
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


    def isExist(self, ind, EP):   #判断个体ind的适应度是否与EP中某个个体的适应度相对，若相等，则返回True
        for ep in EP:
            if ind.fitness == ep.fitness: # 判断两个列表对应元素的值是否相等
                return True
        return False


    def isEP_Dominated_ind(self, ind, EP):  # 判断EP中的某个个体是否支配ind，若支配，则返回True
        for ep in EP:
            if self.isDominated(ep.fitness, ind.fitness):
                return True
        return False




    def calculateScheduleEnergy(self, schedule):
        schedule.E_total = 0
        for taskId in schedule.taskSet:
            schedule.E_total += schedule.taskSet[taskId].energy




    def printNewSchedule(self, new_schedule):
        print("S_new--", new_schedule.S)
        for coreId in new_schedule.S:
            if coreId != 0:
                print("core", coreId, ": ",end="")
                for taskId in new_schedule.S[coreId]:
                    task = new_schedule.taskSet[taskId]
                    print(str(taskId)+"=("+str(task.ST_i_l)+","+str(task.FT_i_l)+") ", end="")
                print("\n")

        for coreId in new_schedule.S:
            if coreId == 0:
                print("WS:      ", end="")
                for taskId in new_schedule.S[coreId]:
                    task = new_schedule.taskSet[taskId]
                    print(str(taskId)+"=("+str(task.ST_i_ws)+","+str(task.FT_i_ws)+") ", end="")
                break
        print("\n")

        for coreId in new_schedule.S:
            if coreId == 0:
                print("Cloud:   ", end="")
                for taskId in new_schedule.S[coreId]:
                    task = new_schedule.taskSet[taskId]
                    print(str(taskId) + "=(" + str(task.ST_i_c) + "," + str(task.FT_i_c) + ") ",end="")
                break
        print("\n")

        for coreId in new_schedule.S:
            if coreId == 0:
                print("WR:      ", end="")
                for taskId in new_schedule.S[coreId]:
                    task = new_schedule.taskSet[taskId]
                    print(str(taskId) + "=(" + str(task.ST_i_wr) + "," + str(task.FT_i_wr) + ") ",end="")
                break
        print("\n")
        print("(Time, Energy)=",new_schedule.fitness)
        print("\n\n")


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

    def getWorkflow(self):
        filename = self.getCurrentPath()+"\GeneralTopology.txt"
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


    def getCurrentPath(self):
        return os.path.dirname(os.path.realpath(__file__))

    def getProjectPath(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(os.path.dirname(cur_path))

    def printPopulationFitness(self, population):
        for ind in population:
            print('Index:  ', population.index(ind), "--",  ind.fitness)



class Workflow:
    def __init__(self):
        self.entryTask = None      #开始任务
        self.exitTask = None       #结束任务
        self.position = []         #执行位置
        self.sequence = []         #执行顺序
        self.taskNumber = None
        self.taskSet = []          #列表的索引值就是任务的id值


class Schedule:
    def __init__(self):
        self.taskSet = {}
        self.S = {0: [], 1: [], 2: [], 3: []}  # Record the set of task that is executed certain execution unit selection. eg. S[3]=[v1,v3,v5,v7,v9,v10]
        self.coreTP = {1: [0], 2: [0], 3: [0]}  # Index is core number, its element denotes the current time point on the core.
        self.wsTP = [0]  # The current time point on the wireless sending channel.
        self.cloudTP = [0]  # The current time point on the cloud.
        self.wrTP = [0]  # The current time point on the wireless receiving channel.
        self.fitness = []
        self.T_total = None
        self.E_total = 0
        self.rank = None
        self.S_p = []
        self.n = 0


class Task:
    def __init__(self):
        self.T_es = None    # Earliest start time
        self.T_lf = None    # Latest finish time
        self.T_comp = None  # Computation cost
        self.comm = {}      # Communication cost
        self.x_i = None     # Execution decision
        self.w_i = None     # Workload
        self.PCP = []       # Partial critical path
        self.criticalParent = []
        self.temp_criticalParent = None

        self.id = None
        self.preTaskSet = []  #The set of predecessor task (element is Task class).
        self.sucTaskSet = []  #The set of successor task (element is Task class).
        self.isSchedule = False
        self.E_comp = None
        self.E_comm = None

        # self.RT_i_l = None     # The ready time of task vi on a local core.
        # self.RT_i_ws = None    # The ready time of task vi on the wireless sending channel.
        # self.RT_i_c = None     # The ready time of task vi on the cloud.
        # self.RT_i_wr = None    # The ready time for the cloud to transmit back the results of task vi
        #
        # self.ST_i_l = None     # The start time of task vi on a local core.
        # self.ST_i_ws = None    # The start time of task vi on the wireless sending channel.
        # self.ST_i_c = None     # The start time of task vi on the cloud.
        # self.ST_i_wr = None    # The start time for the cloud to transmit back the results of task vi
        #
        # self.FT_i_l = None     # The finish time of task vj on a local core.
        # self.FT_i_ws = None    # The finish time of task vj on the wireless sending channel.
        # self.FT_i_c = None     # The finish time of task vj on the cloud.
        # self.FT_i_wr = None    # The finish time of task vj on the wireless receiving channel.
        # self.energy = 0
