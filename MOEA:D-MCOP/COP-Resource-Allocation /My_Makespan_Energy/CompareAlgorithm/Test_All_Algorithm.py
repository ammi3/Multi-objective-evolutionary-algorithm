import pandas as pd

from My_Makespan_Energy.CompareAlgorithm import NSGA2, MOWOA, MOFOA, iMABC, MOEAD, MOEAD_DVFS, DVFSA_2, HGPCA, CTESA, eDors
import copy
import os
import time
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import xlwt


class Test_All_Algorithm:
    # def __init__(self):
    #     algorithmName_list = ["NSGA2", "DVFS", "MOEAD", "MOEAD_DVFS"]
    #     for TN in range(10, 101, 10):
    #         self.getReferPF(TN, algorithmName_list)

    def getIGD_EA(self, algorithmName_list, runTime, taskNumberRange):
        readAlgorithmPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        readReferPFPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        IGDFilePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"+"IGD.xls"
        data = xlrd.open_workbook(readReferPFPath)
        table = data.sheet_by_name('total')
        PF_ref = table._cell_values

        f = xlwt.Workbook(IGDFilePath)
        IGD_mean_std_total_list = []
        for alg in algorithmName_list:
            sheet = f.add_sheet(alg)
            filePath = readAlgorithmPath + alg + ".xls"
            data = xlrd.open_workbook(filePath)
            if alg != 'DVFS':
                IGDValue_list = []
                for i in range(0,runTime):
                    table = data.sheet_by_name(str(i+1))
                    PF_know = table._cell_values
                    PF_know.pop() # 删除最后一行的计算时间
                    IGDValue = self.getIGDValue(PF_ref, PF_know)
                    IGDValue_list.append(IGDValue)
                    sheet.write(i, 0, IGDValue)
                mean = float(np.mean(IGDValue_list))
                std = float(np.std(IGDValue_list, ddof=1))
                sheet.write(i+1, 0, mean)
                sheet.write(i+1, 1, std)
                temp = []
                temp.append(mean)
                temp.append(std)
                IGD_mean_std_total_list.append(temp)
            else:
                table = data.sheet_by_name('1')
                PF_know = table._cell_values
                PF_know.pop()  # 删除最后一行的计算时间
                IGDValue = self.getIGDValue(PF_ref, PF_know)
                sheet.write(0, 0, IGDValue)
                temp = []
                temp.append(IGDValue)
                temp.append(0)
                IGD_mean_std_total_list.append(temp)

        sheet_IGD_total = f.add_sheet('IGD-total')
        i=1
        for IGD_mean_std in IGD_mean_std_total_list:
            sheet_IGD_total.write(i, 0, round(IGD_mean_std[0], 2))
            sheet_IGD_total.write(i, 1, round(IGD_mean_std[1], 2))
            i+=1
        f.save(IGDFilePath)


    def getIGD(self, algorithmName_list, runTime, taskNumberRange, algorithmType):
        readAlgorithmPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        readReferPFPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        IGDFilePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"+"IGD_"+algorithmType+".xls"
        data = xlrd.open_workbook(readReferPFPath)
        table = data.sheet_by_name('total')
        PF_ref = table._cell_values

        f = xlwt.Workbook(IGDFilePath)
        IGD_mean_std_total_list = []
        for alg in algorithmName_list:
            sheet = f.add_sheet(alg)
            filePath = readAlgorithmPath + alg + ".xls"
            data = xlrd.open_workbook(filePath)
            if alg not in ['TSDVFS', 'CTESA', 'eDors']:
                IGDValue_list = []
                for i in range(0,runTime):
                    table = data.sheet_by_name(str(i+1))
                    PF_know = table._cell_values
                    PF_know.pop() # 删除最后一行的计算时间
                    IGDValue = self.getIGDValue(PF_ref, PF_know)
                    IGDValue_list.append(IGDValue)
                    sheet.write(i, 0, IGDValue)
                mean = float(np.mean(IGDValue_list))
                std = float(np.std(IGDValue_list, ddof=1))
                sheet.write(i+1, 0, mean)
                sheet.write(i+1, 1, std)
                temp = []
                temp.append(mean)
                temp.append(std)
                IGD_mean_std_total_list.append(temp)
            else:
                table = data.sheet_by_name('1')
                PF_know = table._cell_values
                PF_know.pop()  # 删除最后一行的计算时间
                IGDValue = self.getIGDValue(PF_ref, PF_know)
                sheet.write(0, 0, IGDValue)
                temp = []
                temp.append(IGDValue)
                temp.append(0)
                IGD_mean_std_total_list.append(temp)

        sheet_IGD_total = f.add_sheet('IGD-total')
        i=1
        for IGD_mean_std in IGD_mean_std_total_list:
            sheet_IGD_total.write(i, 0, round(IGD_mean_std[0], 2))
            sheet_IGD_total.write(i, 1, round(IGD_mean_std[1], 2))
            i+=1
        f.save(IGDFilePath)


    def getIGDValue(self, PF_ref, PF_know):
        sum = []
        for v in PF_ref:
            distance = self.d_v_PFSet(v, PF_know)
            sum.append(distance)
        return np.average(sum)


    def getGD(self, algorithmName_list, runTime, taskNumberRange, algorithmType):
        readAlgorithmPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        readReferPFPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        GDFilePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\" + "GD_"+algorithmType+".xls"
        data = xlrd.open_workbook(readReferPFPath)
        table = data.sheet_by_name('total')
        PF_ref = table._cell_values

        f = xlwt.Workbook(GDFilePath)
        GD_mean_std_total_list = []

        for alg in algorithmName_list:
            sheet = f.add_sheet(alg)
            filePath = readAlgorithmPath + alg + ".xls"
            data = xlrd.open_workbook(filePath)
            if alg not in ['TSDVFS', 'CTESA', 'eDors']:
                GDValue_list = []
                for i in range(0, runTime):
                    table = data.sheet_by_name(str(i + 1))
                    PF_know = table._cell_values
                    PF_know.pop()  # 删除最后一行的计算时间
                    GDValue = self.getGDValue(PF_ref, PF_know)
                    GDValue_list.append(GDValue)
                    sheet.write(i, 0, GDValue)
                mean = float(np.mean(GDValue_list))
                std = float(np.std(GDValue_list, ddof=1))
                sheet.write(i + 1, 0, mean)
                sheet.write(i + 1, 1, std)
                temp = []
                temp.append(mean)
                temp.append(std)
                GD_mean_std_total_list.append(temp)

            else:
                table = data.sheet_by_name('1')
                PF_know = table._cell_values
                PF_know.pop()  # 删除最后一行的计算时间
                GDValue = self.getGDValue(PF_ref, PF_know)
                sheet.write(0, 0, GDValue)
                temp = []
                temp.append(GDValue)
                temp.append(0)
                GD_mean_std_total_list.append(temp)
        sheet_GD_total = f.add_sheet('GD-total')
        i = 1
        for GD_mean_std in GD_mean_std_total_list:
            sheet_GD_total.write(i, 0, round(GD_mean_std[0], 2))
            sheet_GD_total.write(i, 1, round(GD_mean_std[1], 2))
            i += 1
        f.save(GDFilePath)


    def getGDValue(self, PF_ref, PF_know):
        sum = []
        for v in PF_know:
            distance = self.d_v_PFSet(v, PF_ref)
            sum.append(distance)
        return np.sqrt(np.average(sum))


    def getMS(self, algorithmName_list, runTime, taskNumberRange):
        readAlgorithmPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        readReferPFPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        MSFilePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\" + "MS.xls"
        data = xlrd.open_workbook(readReferPFPath)
        table = data.sheet_by_name('total')
        PF_ref = table._cell_values

        f = xlwt.Workbook(MSFilePath)
        MS_mean_std_total_list = []
        for alg in algorithmName_list:
            sheet = f.add_sheet(alg)
            filePath = readAlgorithmPath + alg + ".xls"
            data = xlrd.open_workbook(filePath)
            if alg != 'DVFS':
                MSValue_list = []
                for I in range(0, runTime):
                    table = data.sheet_by_name(str(I + 1))
                    PF_know = table._cell_values
                    PF_know.pop()  # 删除最后一行的计算时间
                    sum = 0
                    for i in range(2):
                        if i == 0:
                            fz_1 = min(PF_know[-1][i], PF_ref[-1][i])
                            fz_2 = max(PF_know[0][i], PF_ref[0][i])
                            fm = PF_ref[-1][i] - PF_ref[0][i]
                            sum += pow((fz_1 - fz_2) / fm, 2)
                        else:
                            fz_1 = min(PF_know[0][i], PF_ref[0][i])
                            fz_2 = max(PF_know[-1][i], PF_ref[-1][i])
                            fm = PF_ref[0][i] - PF_ref[-1][i]
                            sum += pow((fz_1 - fz_2) / fm, 2)
                    MSValue_list.append(np.sqrt(sum / 2))
                    sheet.write(I, 0, np.sqrt(sum / 2))
                mean = float(np.mean(MSValue_list))
                std = float(np.std(MSValue_list, ddof=1))
                sheet.write(I + 1, 0, mean)
                sheet.write(I + 1, 1, std)
                temp = []
                temp.append(mean)
                temp.append(std)
                MS_mean_std_total_list.append(temp)
            else:
                table = data.sheet_by_name('1')
                PF_know = table._cell_values
                PF_know.pop()  # 删除最后一行的计算时间
                sum = 0
                for i in range(2):
                    if i == 0:
                        fz_1 = min(PF_know[-1][i], PF_ref[-1][i])
                        fz_2 = max(PF_know[0][i], PF_ref[0][i])
                        fm = PF_ref[-1][i] - PF_ref[0][i]
                        sum += pow((fz_1 - fz_2) / fm, 2)
                    else:
                        fz_1 = min(PF_know[0][i], PF_ref[0][i])
                        fz_2 = max(PF_know[-1][i], PF_ref[-1][i])
                        fm = PF_ref[0][i] - PF_ref[-1][i]
                        sum += pow((fz_1 - fz_2) / fm, 2)
                mean = float(np.sqrt(sum / 2))
                sheet.write(0, 0, mean)
                temp = []
                temp.append(mean)
                temp.append(0)
                MS_mean_std_total_list.append(temp)
        sheet_MS_total = f.add_sheet('MS-total')
        i = 1
        for MS_mean_std in MS_mean_std_total_list:
            sheet_MS_total.write(i, 0, round(MS_mean_std[0], 2))
            sheet_MS_total.write(i, 1, round(MS_mean_std[1], 2))
            i += 1
        f.save(MSFilePath)


    def d_v_PFSet(self, v, PFSet):  # 求v和PFSet中最近的距离
        dList = []
        for pf in PFSet:
            distance = self.getDistance(v, pf)
            dList.append(distance)
        return min(dList)


    def getDistance(self, point1, point2):
        return np.sqrt(np.sum(np.square([point1[i] - point2[i] for i in range(2)])))


    def update_EP_History(self, EP_Current, EP_History):  # 用当前运行后的非支配解集EP_Current来更新历史非支配解集EP_History
        if (EP_History == []):
            for epc in EP_Current:
                EP_History.append(copy.copy(epc))
        else:
            for epc in EP_Current:
                if (self.isExist(epc, EP_History) == False):  # 先判断ep是否在EP_History中，若不在，则返回False。
                    if (self.isEP_Dominated_ind(EP_History, epc) == False):  # 然后再判断EP_History是否支配ep
                        i = 0
                        while (i < EP_History.__len__()):  # 判断ep是否支配EP中的非支配解，若支配，则删除它所支配的解
                            if (self.isDominated(epc, EP_History[i]) == True):
                                EP_History.remove(EP_History[i])
                                i -= 1
                            i += 1
                        EP_History.append(copy.copy(epc))


    def isExist(self, ep, EP_History):   #判断ep是否与EP中某个支配解相对，若相等，则返回True
        for eph in EP_History:
            if ep == eph: # 判断两个列对应元素的值是否相等
                return True
        return False


    def isEP_Dominated_ind(self, EP_History, ep):   #判断EP中的某个非支配解是否支配ep，若支配，则返回True
        for eph in EP_History:
            if self.isDominated(eph, ep):
                return True
        return False


    def isDominated(self, fitness_1, fitness_2):  # 前者是否支配后者
        flag = -1
        for i in range(2):
            if fitness_1[i] < fitness_2[i]:
                flag = 0
            if fitness_1[i] > fitness_2[i]:
                return False
        if flag == 0:
            return True
        else:
            return False


    def getReferParetoFront(self, algorithmName_list, taskNumberRange):
        readPath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\"
        writePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\referPF.xls"
        referPF = []
        for alg in algorithmName_list:
            filePath = readPath + alg+".xls"
            data = xlrd.open_workbook(filePath)
            table = data.sheet_by_name('total')
            currentPF = table._cell_values
            currentPF.pop() #删除最后一行的运行时间
            self.update_EP_History(currentPF, referPF)

        referPF = sorted(referPF, key=itemgetter(0))
        f = xlwt.Workbook(writePath)
        sheet = f.add_sheet('total')
        for i in range(len(referPF)):
            sheet.write(i, 0, referPF[i][0])
            sheet.write(i, 1, referPF[i][1])
        f.save(writePath)


    def writeEPToExcelFile(self, f, EP, sheetName, computationTime):
        newEP = sorted(EP, key=itemgetter(0))
        sheet = f.add_sheet(sheetName)
        for i in range(len(newEP)):
            sheet.write(i, 0, newEP[i][0])
            sheet.write(i, 1, newEP[i][1])
        sheet.write(i+1, 0, computationTime)

    def write_IGD_list_Current_ToExcelFile(self, f_IGDCurve, IGD_list_Current, sheetName):
        sheet = f_IGDCurve.add_sheet(sheetName)
        for i in range(len(IGD_list_Current)):
            sheet.write(i, 0, IGD_list_Current[i])


    def getCurrentPath(self):
        return os.path.dirname(os.path.realpath(__file__))


    def getProjectPath(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(os.path.dirname(cur_path))


    def getreferPFFromFile(self, taskNumber):
        filename = self.getProjectPath()+'\My_Makespan_Energy\ExperimentResult\\'+'referPF.xls'
        data = xlrd.open_workbook(filename)
        table = data.sheet_by_name('total')
        PF = table._cell_values
        PF = np.array(PF)
        return (PF[:, 0], PF[:, 1])


    def getParetoFront(self, algorithmName, taskNumberRange):
        filename = self.getCurrentPath()+'\ExperimentResult\\'+taskNumberRange+'\\'+algorithmName+'.xls'
        data = xlrd.open_workbook(filename)
        table = data.sheet_by_name('total')
        PF = table._cell_values
        PF.pop()
        PF = np.array(PF)
        return (PF[:,0], PF[:,1])


    def plot_EA_ParetoFront(self, algorithmName_list, taskNumberRange):
        XY = []
        FG = []
        for algorithmName in algorithmName_list:
            (x, y) = self.getParetoFront(algorithmName, taskNumberRange)
            XY.append((x, y))
        Marker = ['+', 'o', '<', 'v', 'x', 'd']
        Color  = ['m', 'y', 'g', 'c', 'r', 'k']
        for i in range(len(XY)):
            if algorithmName_list[i]=="NSGA2":
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label="NSGA-II")
                FG.append(fg)
            elif algorithmName_list[i]=="MOEAD":
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label="MOEA/D")
                FG.append(fg)
            elif algorithmName_list[i]=="MOEAD-DVFS":
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label="MOEA/D-MCOP")
                FG.append(fg)
            else:
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label=algorithmName_list[i])
                FG.append(fg)
        # font={'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        font={'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        plt.legend(handles=FG, prop=font, loc='lower right')
        plt.xlabel('ACT (Sec.)', font)
        plt.ylabel('AEC (J)', font)
        fig = plt.gcf()
        filename = self.getCurrentPath() + "\ExperimentResult\\" +taskNumberRange+ '\PFCurve_MOEA'
        fig.savefig(filename, figsize=(1,1), dpi=800)
        fig.clear()


    def plot_Heuristic_ParetoFront(self, algorithmName_list, taskNumberRange):
        XY = []
        FG = []
        for algorithmName in algorithmName_list:
            (x, y) = self.getParetoFront(algorithmName, taskNumberRange)
            XY.append((x, y))
        Marker = ['*', 'o', 'x', 'd']
        Color = ['g', 'b','r','k']
        for i in range(len(XY)):
            if algorithmName_list[i]=="MOEAD-DVFS":
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label="MOEA/D-MCOP")
                FG.append(fg)
            else:
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label=algorithmName_list[i])
                FG.append(fg)
        # font={'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        font={'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        plt.legend(handles=FG, prop=font, loc='lower right')
        plt.xlabel('ACT (Sec.)', font)
        plt.ylabel('AEC (J)', font)
        fig = plt.gcf()
        filename = self.getCurrentPath() + "\ExperimentResult\\" +taskNumberRange+ '\PFCurve_Heuristic'
        fig.savefig(filename, figsize=(1,1), dpi=800)
        fig.clear()

    # Test.plot_All_ParetoFront(['NSGA2', 'MOWOA', 'MOFOA', 'HGPCA', 'MOEAD', 'TSDVFS', 'CTESA', 'eDors', 'MOEAD-DVFS'], taskNumberRange)

    def plot_All_ParetoFront(self, algorithmName_list, taskNumberRange):
        XY = []
        FG = []
        for algorithmName in algorithmName_list:
            (x, y) = self.getParetoFront(algorithmName, taskNumberRange)
            XY.append((x, y))
        Marker = ['+', 'o', '<', 'v', 'x', '^', '>', '*', 'd']
        Color  = ['m', 'y', 'g', 'c', 'r', 'indigo', 'sienna', 'tan', 'k']
        for i in range(len(XY)):
            if algorithmName_list[i]=="NSGA2":
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label="NSGA-II")
                FG.append(fg)
            elif algorithmName_list[i]=="MOEAD":
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label="MOEA/D")
                FG.append(fg)
            elif algorithmName_list[i]=="MOEAD-DVFS":
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label="MOEA/D-STGO")
                FG.append(fg)
            else:
                fg, = plt.plot(XY[i][0], XY[i][1], marker=Marker[i], markersize=3, color=Color[i], linestyle='',linewidth=2, label=algorithmName_list[i])
                FG.append(fg)
        # font={'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        font={'size': 10}
        plt.legend(handles=FG, prop=font, loc='lower right')
        plt.xlabel('AED (Sec.)', font)
        plt.ylabel('AEC (J)', font)
        fig = plt.gcf()
        filename = self.getCurrentPath() + "\ExperimentResult\\" +taskNumberRange+ '\All_PFcurve.pdf'
        fig.savefig(filename, figsize=(1,1), bbox_inches='tight')
        fig.clear()


    def getEPSet_FT_EC(self, algorithmName, taskNumberRange):
        filePath = self.getCurrentPath()+'\ExperimentResult\\'+taskNumberRange+'\\'+algorithmName+'.xls'
        data = xlrd.open_workbook(filePath)
        table = data.sheet_by_name('total')
        PF = table._cell_values
        PF.pop()
        PF = np.array(PF)
        return PF[:, 0], PF[:, 1]


    def plot_EA_ACT_AEC_BoxGraph(self, algorithmName_list, taskNumberRange):
        completionTime = []
        energyConsumption = []
        for algorithmName in algorithmName_list:
            ctList, ecList = self.getEPSet_FT_EC(algorithmName, taskNumberRange)
            completionTime.append(ctList)
            energyConsumption.append(ecList)

        plt.close()
        font={'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        plt.xlabel('Algorithm', font)
        plt.ylabel('ACT (Sec.)', font)
        temp_algorithmName_list = copy.deepcopy(algorithmName_list)
        temp_algorithmName_list[0] = 'NSGA-II'
        temp_algorithmName_list[-2] = 'MOEA/D'
        temp_algorithmName_list[-1] = 'MOEA/D-MCOP'
        plt.boxplot(completionTime, labels=temp_algorithmName_list)
        filename = self.getCurrentPath() + "\ExperimentResult\\"+taskNumberRange+'\\'+  'PFBox_ACT_MOEA'
        plt.savefig(filename, figsize=(1, 1), dpi=800)

        plt.close()
        plt.boxplot(energyConsumption, labels=temp_algorithmName_list)
        plt.xlabel('Algorithm', font)
        plt.ylabel('AEC (J)', font)
        filename = self.getCurrentPath() + "\ExperimentResult\\" +taskNumberRange+'\\'+ 'PFBox_AEC_MOEA'
        plt.savefig(filename, figsize=(1, 1), dpi=800)
        plt.close()


    def plot_Heuristic_ACT_AEC_BoxGraph(self, algorithmName_list, taskNumberRange):
        completionTime = []
        energyConsumption = []
        for algorithmName in algorithmName_list:
            ctList, ecList = self.getEPSet_FT_EC(algorithmName, taskNumberRange)
            completionTime.append(ctList)
            energyConsumption.append(ecList)

        plt.close()
        font={'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        plt.xlabel('Algorithm', font)
        plt.ylabel('ACT (Sec.)', font)
        temp_algorithmName_list = copy.deepcopy(algorithmName_list)
        temp_algorithmName_list[-1] = 'MOEA/D-MCOP'
        plt.boxplot(completionTime, labels=temp_algorithmName_list)
        filename = self.getCurrentPath() + "\ExperimentResult\\"+taskNumberRange+'\\'+  'PFBox_ACT_Heuristic'
        plt.savefig(filename, figsize=(1, 1), dpi=800)

        plt.close()
        plt.boxplot(energyConsumption, labels=temp_algorithmName_list)
        plt.xlabel('Algorithm', font)
        plt.ylabel('AEC (J)', font)
        filename = self.getCurrentPath() + "\ExperimentResult\\" +taskNumberRange+'\\'+ 'PFBox_AEC_Heuristic'
        plt.savefig(filename, figsize=(1, 1), dpi=800)
        plt.close()

        # Test.plot_All_ACT_AEC_BoxGraph(['NSGA2', 'MOWOA', 'MOFOA', 'HGPCA', 'MOEAD', 'TSDVFS', 'CTESA', 'eDors', 'MOEAD-DVFS'], taskNumberRange)


    def plot_All_ACT_AEC_BoxGraph(self, algorithmName_list, taskNumberRange):
        completionTime = []
        energyConsumption = []
        for algorithmName in algorithmName_list:
            ctList, ecList = self.getEPSet_FT_EC(algorithmName, taskNumberRange)
            completionTime.append(ctList)
            energyConsumption.append(ecList)

        plt.close()
        font = { 'size': 10}
        plt.xlabel('Algorithm', font)
        plt.ylabel('AED (Sec.)', font)
        temp_algorithmName_list = copy.deepcopy(algorithmName_list)
        temp_algorithmName_list[0] = 'NSGA-II'
        temp_algorithmName_list[4] = 'MOEA/D'
        temp_algorithmName_list[-1] = 'MOEA/D-STGO'
        plt.boxplot(completionTime,
                    labels=temp_algorithmName_list,
                    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
                    meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4},)
        plt.xticks(rotation=18)
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + '\\' + 'PFBox_AED_All.pdf'
        plt.savefig(filename, figsize=(1, 1), bbox_inches = 'tight')


        plt.close()
        plt.boxplot(energyConsumption,
                    labels=temp_algorithmName_list,
                    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
                    meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4}, )
        plt.xticks(rotation=18)
        plt.xlabel('Algorithm', font)
        plt.ylabel('AEC (J)', font)
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + '\\' + 'PFBox_AEC_All.pdf'
        plt.savefig(filename, figsize=(1, 1), bbox_inches = 'tight')
        plt.close()


    def get_IGD_GD_MS_Value(self, algorithmName, taskNumberRange):
        IGD_filePath = self.getCurrentPath() + '\ExperimentResult\\' + taskNumberRange + '\\' + 'IGD.xls'
        data = xlrd.open_workbook(IGD_filePath)
        table = data.sheet_by_name(algorithmName)
        IGD = table._cell_values
        IGD.pop()
        IGD = np.array(IGD)
        IGD = IGD[:, 0]
        IGD = [float(igd) for igd in IGD]

        GD_filePath = self.getCurrentPath() + '\ExperimentResult\\' + taskNumberRange + '\\' + 'GD.xls'
        data = xlrd.open_workbook(GD_filePath)
        table = data.sheet_by_name(algorithmName)
        GD = table._cell_values
        GD.pop()
        GD = np.array(GD)
        GD = GD[:, 0]
        GD = [float(gd) for gd in GD]

        MS_filePath = self.getCurrentPath() + '\ExperimentResult\\' + taskNumberRange + '\\' + 'MS.xls'
        data = xlrd.open_workbook(MS_filePath)
        table = data.sheet_by_name(algorithmName)
        MS = table._cell_values
        MS.pop()
        MS = np.array(MS)
        MS = MS[:, 0]
        MS = [float(ms) for ms in MS]
        return IGD, GD, MS


    def plot_IGD_GD_MS_BoxGraph(self, algorithmName_list, taskNumberRange):
        IGD_list = []
        GD_list = []
        MS_list = []
        for algorithmName in algorithmName_list:
            IGD, GD, MS = self.get_IGD_GD_MS_Value(algorithmName, taskNumberRange)
            IGD_list.append(IGD)
            GD_list.append(GD)
            MS_list.append(MS)

        plt.close()
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        plt.xlabel('Algorithm', font)
        plt.ylabel('IGD', font)
        plt.boxplot(IGD_list, labels=['NSGA-II','MOWOA','MOFOA','iMABC', 'MOEA/D', 'MOEA/D-MCOP'])
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + '\Box_IGD'
        plt.savefig(filename, figsize=(1, 1), dpi=800)

        plt.close()
        plt.boxplot(GD_list, labels=['NSGA-Ⅱ','MOWOA','MOFOA','iMABC', 'MOEA/D', 'MOEA/D-MCOP'])
        plt.xlabel('Algorithm', font)
        plt.ylabel('GD', font)
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + '\Box_GD'
        plt.savefig(filename, figsize=(1, 1), dpi=800)
        plt.close()

        plt.close()
        plt.boxplot(MS_list, labels=['NSGA-Ⅱ','MOWOA','MOFOA','iMABC', 'MOEA/D', 'MOEA/D-MCOP'])
        plt.xlabel('Algorithm', font)
        plt.ylabel('MS', font)
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + '\Box_MS'
        plt.savefig(filename, figsize=(1, 1), dpi=800)
        plt.close()


    def plot_IGD_BoxGraph(self, algorithmName_list, taskNumberRange, algorithmType):
        IGD_list = []
        GD_list = []
        MS_list = []
        for algorithmName in algorithmName_list:
            IGD, GD, MS = self.get_IGD_GD_MS_Value(algorithmName, taskNumberRange)
            IGD_list.append(IGD)

        plt.close()
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        plt.xlabel('Algorithm', font)
        plt.ylabel('IGD', font)
        plt.boxplot(IGD_list, labels=['TSDVFA','CTESA','eDors', 'MOEA/D-MCOP'])
        filename = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + '\Box_IGD_'+algorithmType
        plt.savefig(filename, figsize=(1, 1), dpi=800)


    def getAlgorithm_ACT(self, algorithmName, taskNumberRange):
        filePath = self.getCurrentPath()+'\ExperimentResult\\'+taskNumberRange+'\\'+algorithmName+'.xls'
        data = xlrd.open_workbook(filePath)
        table = data.sheet_by_name('total')
        PF = table._cell_values
        PF = np.array(PF)
        return float(PF[-1][0])


    def getACT(self, algorithmName_list, taskNumberRange):
        writePath = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\ACT.xls"
        f = xlwt.Workbook(writePath)
        sheet = f.add_sheet('ACT')
        for i in range(len(algorithmName_list)):
            alg_ACT = self.getAlgorithm_ACT(algorithmName_list[i], taskNumberRange)
            sheet.write(0, i, round(alg_ACT, 2))
        f.save(writePath)


    def plot_IGD_ConvergenceCurve(self, algorithmName, taskNumberRange, IGD_list_History):
        generation_list = [i for i in range(1, 101)]
        plt.close()
        Color = ['g', 'b','r','k']
        FG = []
        fg, = plt.plot(generation_list, IGD_list_History, color=Color[3], linestyle='-', linewidth=2, label=taskNumberRange)
        FG.append(fg)

        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
        plt.legend(handles=FG, prop=font, loc='lower right')
        plt.xlabel('Generation', font)
        plt.ylabel('IGD', font)
        fig = plt.gcf()
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\IGDConvergenceCurve_" + algorithmName
        fig.savefig(filename, figsize=(1, 1), dpi=800)
        fig.clear()


    def update_IGD_list_History(self, IGD_list_Current, IGD_list_History):
        if IGD_list_History == []:
            for IGD in IGD_list_Current:
                IGD_list_History.append(IGD)
        else:
            for i in range(len(IGD_list_Current)):
                IGD_list_History[i] += IGD_list_Current[i]


    def NSGA2_Run(self, popSize, maxGen, pc, pm_SMD, pm_bit, runTime, taskNumberRange):
        print("*** NSGA2 (Run "+taskNumberRange+' '+str(runTime)+" time) ***")
        EP_History = []
        time_list = []
        filename_PF = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\NSGA2.xls"
        f_PF = xlwt.Workbook(filename_PF)

        IGD_list_History = []
        filename_IGDList = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\NSGA2_IGDCurve.xls"
        f_IGDList = xlwt.Workbook(filename_IGDList)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            nsga2 = NSGA2.NSGA2(popSize, maxGen, pc, pm_SMD, pm_bit, taskNumberRange)
            EP_Current, IGD_list_Current = nsga2.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            time_list.append(CT)
            self.writeEPToExcelFile(f_PF, EP_Current, str(I), CT)
            self.update_EP_History(EP_Current, EP_History)

            self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_Current, str(I))
            self.update_IGD_list_History(IGD_list_Current, IGD_list_History)

        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f_PF, EP_History, 'total', np.average(time_list))
        f_PF.save(filename_PF)

        IGD_list_History = [IGD / runTime for IGD in IGD_list_History]
        self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_History, 'total')
        f_IGDList.save(filename_IGDList)


    def MOWOA_Run(self,popSize, maxGen, runTime, taskNumberRange):
        print("*** MOWOA (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename_PF = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOWOA.xls"
        f_PF = xlwt.Workbook(filename_PF)

        IGD_list_History = []
        filename_IGDList = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\MOWOA_IGDCurve.xls"
        f_IGDList = xlwt.Workbook(filename_IGDList)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            woa = MOWOA.WOA(popSize, maxGen, taskNumberRange)
            EP_Current, IGD_list_Current = woa.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f_PF, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)

            self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_Current, str(I))
            self.update_IGD_list_History(IGD_list_Current, IGD_list_History)

        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f_PF, EP_History, 'total', np.average(time_list))
        f_PF.save(filename_PF)

        IGD_list_History = [IGD / runTime for IGD in IGD_list_History]
        self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_History, 'total')
        f_IGDList.save(filename_IGDList)


    def MOFOA_Run(self, popSize, maxGen, runTime, taskNumberRange):
        print("*** MOFOA (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename_PF = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOFOA.xls"
        f_PF = xlwt.Workbook(filename_PF)

        IGD_list_History = []
        filename_IGDList = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\MOFOA_IGDCurve.xls"
        f_IGDList = xlwt.Workbook(filename_IGDList)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            foa = MOFOA.MOFOA(popSize, maxGen, taskNumberRange)
            EP_Current, IGD_list_Current = foa.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f_PF, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)

            self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_Current, str(I))
            self.update_IGD_list_History(IGD_list_Current, IGD_list_History)

        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f_PF, EP_History, 'total', np.average(time_list))
        f_PF.save(filename_PF)

        IGD_list_History = [IGD / runTime for IGD in IGD_list_History]
        self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_History, 'total')
        f_IGDList.save(filename_IGDList)


    def iMABC_Run(self, popSize, maxGen, runTime, taskNumberRange):
        print("*** iMABC (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename_PF = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\iMABC.xls"
        f_PF = xlwt.Workbook(filename_PF)

        IGD_list_History = []
        filename_IGDList = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\iMABC_IGDCurve.xls"
        f_IGDList = xlwt.Workbook(filename_IGDList)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            imabc = iMABC.iMABC(popSize, maxGen, taskNumberRange)
            EP_Current, IGD_list_Current = imabc.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f_PF, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)

            self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_Current, str(I))
            self.update_IGD_list_History(IGD_list_Current, IGD_list_History)

        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f_PF, EP_History, 'total', np.average(time_list))
        f_PF.save(filename_PF)

        IGD_list_History = [IGD / runTime for IGD in IGD_list_History]
        self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_History, 'total')
        f_IGDList.save(filename_IGDList)


    def HGPCA_Run(self, popSize, maxGen, runTime, taskNumberRange):
        print("*** HGPCA (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename_PF = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\HGPCA.xls"
        f_PF = xlwt.Workbook(filename_PF)

        IGD_list_History = []
        filename_IGDList = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\HGPCA_IGDCurve.xls"
        f_IGDList = xlwt.Workbook(filename_IGDList)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            hgpca = HGPCA.HGPCA(popSize, maxGen, pc, pm_SMD, pm_bit, taskNumberRange)
            EP_Current, IGD_list_Current = hgpca.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f_PF, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)

            self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_Current, str(I))
            self.update_IGD_list_History(IGD_list_Current, IGD_list_History)

        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f_PF, EP_History, 'total', np.average(time_list))
        f_PF.save(filename_PF)

        IGD_list_History = [IGD / runTime for IGD in IGD_list_History]
        self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_History, 'total')
        f_IGDList.save(filename_IGDList)


    def MOEAD_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename_PF = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOEAD.xls"
        f_PF = xlwt.Workbook(filename_PF)

        IGD_list_History = []
        filename_IGDList = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\MOEAD_IGDCurve.xls"
        f_IGDList = xlwt.Workbook(filename_IGDList)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            moead = MOEAD.MOEAD(popSize, maxGen, T, taskNumberRange)
            EP_Current, IGD_list_Current = moead.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f_PF, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)

            self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_Current, str(I))
            self.update_IGD_list_History(IGD_list_Current, IGD_list_History)

        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f_PF, EP_History, 'total', np.average(time_list))
        f_PF.save(filename_PF)

        IGD_list_History = [IGD / runTime for IGD in IGD_list_History]
        self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_History, 'total')
        f_IGDList.save(filename_IGDList)


    def MOEAD_DVFS_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** MOEAD_DVFS (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename_PF = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\MOEAD-DVFS.xls"
        f_PF = xlwt.Workbook(filename_PF)

        IGD_list_History = []
        filename_IGDList = self.getCurrentPath() + "\ExperimentResult\\" + taskNumberRange + "\\MOEAD-DVFS_IGDCurve.xls"
        f_IGDList = xlwt.Workbook(filename_IGDList)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            moead_svfs = MOEAD_DVFS.MOEAD_DVFS(popSize, maxGen, T, taskNumberRange)
            EP_Current, IGD_list_Current = moead_svfs.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            time_list.append(CT)
            self.writeEPToExcelFile(f_PF, EP_Current, str(I), CT)
            self.update_EP_History(EP_Current, EP_History)

            self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_Current, str(I))
            self.update_IGD_list_History(IGD_list_Current, IGD_list_History)

        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f_PF, EP_History, 'total', np.average(time_list))
        f_PF.save(filename_PF)

        IGD_list_History = [IGD / runTime for IGD in IGD_list_History]
        self.write_IGD_list_Current_ToExcelFile(f_IGDList, IGD_list_History, 'total')
        f_IGDList.save(filename_IGDList)


    def DVFSA_Run(self, popSize, maxGen, T, runTime, taskNumberRange):
        print("*** DVFSA (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\DVFS.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            svfs = DVFSA_2.DVFSA(popSize, maxGen, T, taskNumberRange)
            EP_Current = svfs.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)


    def CTESA_Run(self, popSize, maxGen, runTime, taskNumberRange, Td_Range):
        print("*** CTESA (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\CTESA.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            ctesa = CTESA.CTESA(popSize, maxGen, taskNumberRange, Td_Range)
            EP_Current = ctesa.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)


    def eDors_Run(self, popSize, maxGen, runTime, taskNumberRange, Td_Range):
        print("*** eDors (Run " +taskNumberRange+' '+ str(runTime) + " time) ***")
        EP_History = []
        time_list = []
        filename = self.getCurrentPath()+"\ExperimentResult\\"+taskNumberRange+"\\eDors.xls"
        f = xlwt.Workbook(filename)
        for I in range(1, runTime + 1):
            print("The " + str(I) + "-th time")
            startTime = time.time()
            edors = eDors.eDors(popSize, maxGen, taskNumberRange, Td_Range)
            EP_Current = edors.run()
            endTime = time.time()
            CT = endTime - startTime
            print("Computation time: ", CT)
            self.writeEPToExcelFile(f, EP_Current, str(I), CT)
            time_list.append(CT)
            self.update_EP_History(EP_Current, EP_History)
        print("ACT: ", np.average(time_list))
        self.writeEPToExcelFile(f, EP_History, 'total', np.average(time_list))
        f.save(filename)


    def plot_IGD_Convergence_Curve_Proposed_Algorithm(self):
        data = xlrd.open_workbook('../CompareAlgorithm/ExperimentResult/IGDConvergenceCurve_data.xls')
        data = data.sheet_by_name('total')
        data = np.array(data._cell_values)
        size = 13
        font = {'size': size}
        Marker = ['<', '+', 'x', 'v', 'd', '^', '>', '*', 'd']
        Color = ['k', 'g', 'orange', 'b', 'k', 'indigo', 'sienna', 'tan', 'k']
        taskNumberRangeList = ['I-[10,20]', 'I-[15,25]', 'I-[20,30]', 'I-[25,35]', 'I-[30,40]', 'I-[10,40]']

        plt.figure()
        FG = []
        for i in range(6):
            fg, = plt.plot(np.arange(100), data[:, i], label=taskNumberRangeList[i])
            FG.append(fg)

        plt.legend(handles=FG, fontsize=size)
        # plt.title(taskNumberRangeList, fontsize=10)
        plt.tick_params(labelsize=10)
        plt.ylabel('IGD', fontsize=size)
        plt.xlabel('Generation', fontsize=size)
        path = 'F:/Sync-Jianguoyun/博士毕业论文/Dissertation/Figures/IOTJ/' + 'IGDConvergenceCurve.pdf'
        plt.savefig(path, figsize=(2, 1), bbox_inches='tight')
        plt.close()



if __name__=="__main__":
    popSize = 100
    maxGen = 100
    pc = 0.8
    pm_SMD = 0.03
    pm_bit = 0.01
    runTime = 20
    EP_Current_list = []
    EP_History = []
    Td_Range = [6,12]

    taskNumberRangeList = ['[10,20]', '[15,25]','[20,30]', '[25,35]','[30,40]','[10,40]']



    Test = Test_All_Algorithm()
    Test.plot_IGD_Convergence_Curve_Proposed_Algorithm()
    for taskNumberRange in taskNumberRangeList:

        Test.plot_EA_ParetoFront([ 'MOEAD', 'MOEAD-DVFS'], taskNumberRange)



