#-*- coding: UTF-8 -*-

class RulPrediction:    
  
    def update(self, Calinumflag, S, Set1, Set2, T):
        if Calinumflag == 1:                #Calinum变量赋值flag
            Sn.append(S)
            Set1n.append(Set1)
            Set2n.append(Set2)
            Tsn.append(T)
        
    def Ruls(self, Calinum):
        Saver = (initial_S - dead_S)/Default
        if Calinum == 0:
            Ruln[0] = Default - (62 - Sn[0])/Saver - T
            Ruls = Ruln[0]
        if Calinum >= 1:
            Ruln_last[Calinum - 1] = Ruln[Calinum - 1]
            Saver_n[Calinum] = (Sn[0] - Sn[Calinum])/Tsn[Calinum]
            Ruln[Calinum] = Ruln_last[Calinum - 1] - (Saver_n[Calinum] * (T - Tsn[Calinum]))/Saver
            Ruls = Ruln[Calinum]    
        if Ruls <= 0:
            return False  
        return Ruls
    
    def RulSet1(self, Calinum):
        Set1aver = (dead_Set1 - initial_Set1)/Default
        if Calinum == 0:
            Ruln[0] = Default - (Set1n[0] - 10)/Set1aver - T
            RulSet1 = Ruln[0]
        if Calinum >= 1:
            Ruln_last[Calinum - 1] = Ruln[Calinum - 1]
            Set1aver_n[Calinum] = (Set1n[Calinum] - Set1n[0])/Tsn[Calinum]
            Ruln[Calinum] = Ruln_last[Calinum - 1] - (Set1aver_n[Calinum] * (T - Tsn[Calinum]))/Set1aver
            RulSet1 = Ruln[Calinum]    
        if RulSet1 <= 0:
            return False  
        return RulSet1
    
    def RulSet2(self, Calinum):
        Set2aver = (dead_Set2 - initial_Set2)/Default
        if Calinum == 0:
            Ruln[0] = Default - (Set2n[0] - 60)/Set2aver - T
            RulSet2 = Ruln[0]
        if Calinum >= 1:
            Ruln_last[Calinum - 1] = Ruln[Calinum - 1]
            Set2aver_n[Calinum] = (Set2n[Calinum] - Set2n[0])/Tsn[Calinum]
            Ruln[Calinum] = Ruln_last[Calinum - 1] - (Set2aver_n[Calinum] * (T - Tsn[Calinum]))/Set2aver
            RulSet2 = Ruln[Calinum]    
        if RulSet2 <= 0:
            return False
        return RulSet2
    
    def Rul(self, Rs, RSet1, RSet2):
        Rul = 0.8*Rs + 0.1*RSet1 + 0.1*RSet2
        return Rul
    


class CaliPrediction:
    
    def NextCaliS(self, Calinum):
        Saver = (initial_S - dead_S)/Default
        if Calinum == 0:
            Calitime_n[0] = deltaS/Saver - T
            CalitimeS = Calitime_n[0]
        if Calinum >= 1:
            Saver_n[Calinum] = (Sn[0] - Sn[Calinum])/Tsn[Calinum]
            Calitime_n[Calinum] = (deltaS  - Saver_n[Calinum] * (T - Tsn[Calinum]) )/Saver
            CalitimeS = Calitime_n[Calinum]
            
        if CalitimeS <= 0:
            return False
        return CalitimeS
    
    def NextCaliSet1(self, Calinum):
        Saver = (dead_Set1 - initial_Set1)/Default
        if Calinum == 0:
            Calitime_n[0] = deltaSet1/Saver - T
            CalitimeSet1 = Calitime_n[0]
        if Calinum >= 1:
            Saver_n[Calinum] = (Sn[0] - Sn[Calinum])/Tsn[Calinum]
            Calitime_n[Calinum] = (deltaSet1  - Saver_n[Calinum] * (T - Tsn[Calinum]) )/Saver
            CalitimeSet1 = Calitime_n[Calinum]
            
        if CalitimeSet1 <= 0:
            return False
        return CalitimeSet1
    
    def NextCaliSet2(self, Calinum):
        Saver = (dead_Set2 - initial_Set2)/Default
        if Calinum == 0:
            Calitime_n[0] = deltaSet2/Saver - T
            CalitimeSet2 = Calitime_n[0]
        if Calinum >= 1:
            Saver_n[Calinum] = (Sn[0] - Sn[Calinum])/Tsn[Calinum]
            Calitime_n[Calinum] = (deltaSet2  - Saver_n[Calinum] * (T - Tsn[Calinum]) )/Saver
            CalitimeSet2 = Calitime_n[Calinum]
            
        if CalitimeSet2 <= 0:
            return False
        return CalitimeSet2
    


class MaintPrediction:

    def NextMaint(self, Rul, Mainnum):
        execute1 = False
        execute2 = False
        if Mainnum == 0:
            if execute1 == False:
                Rul_initialn[0] = Rul
                execute1 = True
            maintime_n[0] = deltaRul - (Rul - Rul_initialn[0])
            Maintime = maintime_n[0]
        if Mainnum >= 1:
            if execute2 == False:
                Rul_initialn[Mainnum] = Rul
                execute2 = True 
            maintime_n[Mainnum] = deltaRul - (Rul - Rul_initialn[Mainnum])
            Maintime = maintime_n[Mainnum]
        
        if Rul < deltaRul:
            Maintime = Rul
        return Maintime
    
if __name__ == '__main__':
    
    #intial 专家经验剩余寿命
    #S
    Default = 400                         #默认电极寿命
    initial_S = 58                          #初始化斜率
    dead_S = 50                          #截止斜率V/pH

    Tsn = [0,30,60,90,120,150]                               #第n次校准时已使用天数                                      （每次校准更新）
    Ruln = [300,250,200,150,100,50]                       #第n次校准到第n+1次校准之前的剩余寿命                （一直更新）
    Ruln_last = [300,250,200,150,100,50]                #第n+1次校准时刻，已减少的剩余寿命                    （每次校准更新）
    Sn = [59,57,56,55,54,53]                                 #第n次校准斜率                                                   （每次校准更新）
    Saver_n = [0.03,0.03,0.03,0.03,0.03,0.03]          #第1次校准到第n次校准之间，斜率平均每天变化量    （每次校准更新）

    #Set1
    initial_Set1 = 10
    dead_Set1 = 40            #mV
    Set1n = [10,18,16,32,38,40]
    Set1aver_n = [0.07,0.07,0.07,0.07,0.07,0.07]

    #Set2
    initial_Set2 = 60
    dead_Set2 = 300          #秒
    Set2n = [60,100,140,180,220,260]
    Set2aver_n = [0.6,0.6,0.6,0.6,0.6,0.6]
    
    #专家经验下次校准
    deltaS = 1
    deltaSet1 = 2
    deltaSet2 = 20

    Sn = [59,57,56,55,54,53]                      #第n次校准斜率                                   （每次校准更新）
    Set1n = [10,18,16,32,38,40]
    Set2n = [60,100,140,180,220,260]
    Calitime_n = [30,30,30,30,30,30]
    
    #专家经验下次维护
    deltaRul = 60
    Rul_initialn = [400, 340, 280, 220, 160, 100]
    maintime_n = [50, 50, 50, 50, 50, 50]
    execute1 = False
    execute2 = False
    
    
