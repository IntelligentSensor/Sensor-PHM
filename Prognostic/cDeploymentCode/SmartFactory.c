/********************************************************************
*文件名:SmartFactory.c
*修改内容：    
    修改函数void SmartFactory_Update(void)
    修改函数TaskSmart
    增加函数void SmartFactory_PentMenuSem(void);
    增加函数void SmartFactory_PostMenuSem(void);
    增加函数void SmartFactory_DeleteMenuTask(void);  将所有删除主函数替换为这个函数
    包括以下函数
        void SmartFactory_CurrentCalStart(INT8U id)
        void SmartFactory_VeiwRunLog(void)
        void SmartFactory_DOElecCal(void)
        void SmartFactory_DOAirCal(void)
        void SmartFactory_DOZeroCal(void)
        void SmartFactory_DOAtmosCal(INT16U atmos)
        void SmartFactory_PHElecCal(void)
        void SmartFactory_PHBufferCal(void)
        void SmartFactory_CondElecCal(void)
        void SmartFactory_TempElecCal(void)
*********************************************************************/
#include "sys_config.h"
#include "SmartFactory.h"

OS_EVENT  *SemSmartFctory;
OS_STK  StkSmart[TASK_SMART_STK_SIZE];//辅助菜单任务堆栈


/*
*********************************************************************************************************
描述：   获取所有界面使用到的信号量
参数：  无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_PentMenuSem(void)
{
    void Modbus_Master_Lock(void);
    INT8U err;
    GUI_X_Lock();
    OSSemPend (SemSPI, 0, &err);
    OSSemPend (SemI2C, 0, &err);
    Modbus_Master_Lock();
}
/*
*********************************************************************************************************
描述：   释放所有界面使用到的信号量
参数：  无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_PostMenuSem(void)
{
    void Modbus_Master_unLock(void);
    GUI_X_Unlock(); 
    OSSemPost (SemSPI);
    OSSemPost (SemI2C); 
    Modbus_Master_unLock();  
}

/*
*********************************************************************************************************
描述：  删除界面任务
参数：  无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_DeleteMenuTask(void)
{
    SmartFactory_PentMenuSem();
    if(OS_ERR_NONE == OSTaskDel (TASK_MENU_PRIO))
    {
        #if OS_TASK_CREATE_EXT_EN > 0u
            OSTaskCreateExt ( TaskMenu,                                          
                       (void *)0, 
                       &StkMenu[TASK_MENU_STK_SIZE-1], 
                       TASK_MENU_PRIO, 
                           TASK_MENU_ID,
                           &StkMenu[0],
                           TASK_MENU_STK_SIZE,
                          (void *)0,                                  
                          OS_TASK_OPT_STK_CHK | OS_TASK_OPT_STK_CLR); 

        #else
            OSTaskCreate ( TaskMenu,                                          
                       (void *)0, 
                       &StkMenu[TASK_MENU_STK_SIZE-1], 
                       TASK_MENU_PRIO );
        #endif 
       

    }    
    SmartFactory_PostMenuSem();
}



/*
*********************************************************************************************************
描述：   智能工厂初始化
参数：  无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_Init(void)
{
    /**********************************
    SYS STATE  初始化
    ************************************/
    INT16U sys_state[4];
  
    sys_state[0] = BOARD_TYPE;
    sys_state[1] = SOFT_VER;
#if(BOARD_TYPE==BOARD_TYPE_DD)
    sys_state[2] = g_stSigConstData.software_ver;
#else
    sys_state[2] = g_stConstData.software_ver;
#endif  
    sys_state[3] = SYS_STATE_INIT;
    WriteModbusSlaveReg(4,sys_state,MAIN_BOARD_TYPE);
  
    SemSmartFctory	= OSSemCreate (0);
#if OS_TASK_CREATE_EXT_EN > 0u
    OSTaskCreateExt ( TaskSmart,                                          
			   (void *)0, 
			   &StkSmart[TASK_SMART_STK_SIZE-1], 
			   TASK_SMART_PRIO, 
				   TASK_SMART_ID,
				   &StkSmart[0],
				   TASK_SMART_STK_SIZE,
				  (void *)0,                                  
                  OS_TASK_OPT_STK_CHK | OS_TASK_OPT_STK_CLR); 

#else
    OSTaskCreate ( TaskSmart,                                          
			   (void *)0, 
			   &StkSmart[TASK_SMART_STK_SIZE-1], 
			   TASK_SMART_PRIO );
#endif 
    
    
    

}


/*
*********************************************************************************************************
描述：   智能工厂  设置继电器
参数：  id  继电器   0-继电器1  1-继电器2
        state   1-打开   0-关闭
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_Relay(INT8U id, INT8U state)
{
    OS_ENTER_CRITICAL();
    g_stFlagVar.HwCtrlRelay[id] = 2;
    OS_EXIT_CRITICAL();
    
    Alarm_SetRelayState(id,state);
}


/*
*********************************************************************************************************
描述：   智能工厂  设置继电器
参数：  id  电流输出   0-电流输出1  1-电流输出2
        value  电流输出值    保留2位小数  如20mA 2000  4mA   400
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_Current(INT8U id, INT16U value)
{
    OS_ENTER_CRITICAL();
    g_stFlagVar.HwCtrlOutput[id] = 2;
    OS_EXIT_CRITICAL();
    g_stAppVar.OutputCur[id] = value/100.0f;
   
}

/*
*********************************************************************************************************
描述：   智能工厂  设置蜂鸣器
参数：  
        state  蜂鸣器状态   1-响  0-不响
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_Beep(INT8U state)
{
    OS_ENTER_CRITICAL();
    g_stMainCfg.Beep = CLOSE;
    OS_EXIT_CRITICAL();
    
    if(state)
    {
        BEEP_OPEN;
    }
    else
    {
        BEEP_CLOSE;
    }
    
}

/*
*********************************************************************************************************
描述：   智能工厂  返回测量界面（主界面）
参数：   无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_ViewMain(void)
{
    
}
//////////////////以下内容需要调用菜单////////////////////////////////////////////

/*
*********************************************************************************************************
描述：   智能工厂  电流校准
参数：   id  电流输出   0-电流输出1  1-电流输出2
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_CurrentCalStart(INT8U id)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
#if(BOARD_TYPE==BOARD_TYPE_DD)
    if(id==0)
        FuncLevel = 0x23;//output1 cal
    else if(id==1)
        FuncLevel = 0x24;//output2 cal
#elif(BOARD_TYPE==BOARD_TYPE_DO)
    if(id==0)
        FuncLevel = 0x26;//output1 cal
    else if(id==1)
        FuncLevel = 0x27;//output2 cal
#elif(BOARD_TYPE==BOARD_TYPE_PH)
    if(id==0)
        FuncLevel = 0x26;//output1 cal
    else if(id==1)
        FuncLevel = 0x27;//output2 cal

#endif
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();
  
}


/*
*********************************************************************************************************
描述：   智能工厂  电流校准结束
参数：   id  电流输出   0-电流输出1  1-电流输出2
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_CurrentCalEnd(INT8U id)
{

    INT32U func_level;  
    INT16U para1;   

    if(id>1)  
        return;
    
    OS_ENTER_CRITICAL();
    func_level = FuncLevel;
    OS_EXIT_CRITICAL();
    
#if(BOARD_TYPE==BOARD_TYPE_DD)
    if(id==0 && func_level != 0x23)
        return;
    else if(id==1 && func_level != 0x24)
        return;
#elif(BOARD_TYPE==BOARD_TYPE_DO)
    if(id==0 && func_level != 0x26)
        return;
    else if(id==1 && func_level != 0x27)
        return;
#elif(BOARD_TYPE==BOARD_TYPE_PH)
    if(id==0 && func_level != 0x26)
        return;
    else if(id==1 && func_level != 0x27)
        return;
#endif
      
    OS_ENTER_CRITICAL();
    para1 = REG_PARA1;
    OS_EXIT_CRITICAL();    
    SmartFactory_VirtualKey_Edit(para1,2000,4,NO_SIGN_MODE);
  
}

/*
*********************************************************************************************************
描述：   智能工厂  察看运行记录
参数：   无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_VeiwRunLog(void)
{
//    INT8U err; 
    OS_ENTER_CRITICAL();
#if(BOARD_TYPE==BOARD_TYPE_DD)
    FuncLevel = 0x51;// run log
#elif(BOARD_TYPE==BOARD_TYPE_DO)
    FuncLevel = 0x51;// run log
#elif(BOARD_TYPE==BOARD_TYPE_PH)
    FuncLevel = 0x61;// run log
#endif
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();
}


/*
*********************************************************************************************************
描述：   智能工厂  溶解氧电气校准
参数：   无
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_DOElecCal(void)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
    FuncLevel = 0x282;//do cal
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask(); 

    SmartFactory_VirtualKey_Enter();

}

/*
*********************************************************************************************************
描述：   智能工厂  溶解氧空气校准
参数：   无
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_DOAirCal(void)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
    FuncLevel = 0x21;//do air cal
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();  

}

/*
*********************************************************************************************************
描述：   智能工厂  溶解氧零点校准
参数：   无
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_DOZeroCal(void)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
    FuncLevel = 0x22;//do zero cal
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();
    
    SmartFactory_VirtualKey_Enter();
}


/*
*********************************************************************************************************
描述：   智能工厂  溶解氧电气校准高倍电压  用于外部校准函数调用
参数：   volt 保留1位小数
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_DOElecCal_x10Volt(FP32 volt)
{
    INT16S tmp[1];
    if(volt>=0)
        tmp[0] = volt*10 + 0.5f;
    else
        tmp[0] = volt*10 - 0.5f;
    
    WriteModbusSlaveReg(1,(INT16U *)tmp,SDO_CALI_TMP_X10V_REG);

}

/*
*********************************************************************************************************
描述：   智能工厂  溶解氧电气校准低倍电压  用于外部校准函数调用
参数：   volt 保留1位小数
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_DOElecCal_x1Volt(FP32 volt)
{
    INT16S tmp[1];
    if(volt>=0)
        tmp[0] = volt*10 + 0.5f;
    else
        tmp[0] = volt*10 - 0.5f;
    
    WriteModbusSlaveReg(1,(INT16U *)tmp,SDO_CALI_TMP_X1V_REG);

}

/*
*********************************************************************************************************
描述：   智能工厂  溶解氧空气校准  用于外部校准函数调用
参数：   参数a  保留2位小数  
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_DOAirCal_a(FP32 a)
{
    INT16S tmp[1];
    if(a>=0)
        tmp[0] = a*100 + 0.5f;
    else
        tmp[0] = a*100 - 0.5f;
    
    WriteModbusSlaveReg(1,(INT16U *)tmp,SDO_CALI_TMP_A_REG);
}

/*
*********************************************************************************************************
描述：   智能工厂  溶解氧零点校准  用于外部校准函数调用
参数：   zero 零点值 保留4位小数
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_DOZeroCal_zero(FP32 zero)
{
    INT16S tmp[1];
    if(zero>=0)
        tmp[0] = zero*10000 + 0.5f;
    else
        tmp[0] = zero*10000 - 0.5f;
    
    WriteModbusSlaveReg(1,(INT16U *)tmp,SDO_CALI_TMP_ZERO_REG);
}



/*
*********************************************************************************************************
描述：   智能工厂  溶解氧大气压校准
参数：   atmos 上位机输入气压值  
返回值： 无
**********************************************************************************************************
*/
#if(BOARD_TYPE==BOARD_TYPE_DO)
void SmartFactory_DOAtmosCal(INT16U atmos)
{
    //INT8U err;
    INT16U meas;
    OS_ENTER_CRITICAL();
    FuncLevel = 0x25;
    if(g_stSigCfg.AtmosCompStyle == MANUAL)
    {
        g_stSigCfg.AtmosCompStyle = AUTO;
        OS_EXIT_CRITICAL();  
        App_SigPara_Save();  
        
        OS_ENTER_CRITICAL();
    }
    OS_EXIT_CRITICAL();  
    
        
    SmartFactory_DeleteMenuTask(); 
    
    OS_ENTER_CRITICAL();
    meas = g_stAppVar.MeasAtmos*10+0.5;
    OS_EXIT_CRITICAL();    
    SmartFactory_VirtualKey_Edit(atmos,meas,4,NO_SIGN_MODE);

  
}
#endif
/*
*********************************************************************************************************
描述：   智能工厂  PH电气校准
参数：   无
返回值： 无
**********************************************************************************************************
*/

void SmartFactory_PHElecCal(void)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
    FuncLevel = 0x282;//ph cal
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();

    SmartFactory_VirtualKey_Enter();
    
    
}
/*
*********************************************************************************************************
描述：   智能工厂  PH缓冲液校准
参数：   无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_PHBufferCal(void)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
    FuncLevel = 0x21;//ph buffer cal
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();

    //SmartFactory_VirtualKey_Enter();
}

/*
*********************************************************************************************************
描述：   智能工厂  电导率电气校准
参数：   无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_CondElecCal(void)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
    FuncLevel = 0x252;//cond cal
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();  

    SmartFactory_VirtualKey_Enter();//将电导电极移开
    
}

/*
*********************************************************************************************************
描述：   智能工厂  电导率电气校准
参数：   无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_TempElecCal(void)
{
    //INT8U err; 
    OS_ENTER_CRITICAL();
#if(BOARD_TYPE==BOARD_TYPE_DD)
    FuncLevel = 0x251;// run log
#elif(BOARD_TYPE==BOARD_TYPE_DO)
    FuncLevel = 0x281;// run log
#elif(BOARD_TYPE==BOARD_TYPE_PH)
    FuncLevel = 0x281;// run log
#endif
    OS_EXIT_CRITICAL();    
        
    SmartFactory_DeleteMenuTask();
    SmartFactory_VirtualKey_Enter();//接第一个电阻

}



/*
*********************************************************************************************************
描述：   智能工厂  电导率设置电极常数
参数：   constant 电极常数    保留4位小数  如0.01  100
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_CondSetCellConstant(FP32 constant)
{
#if(BOARD_TYPE==BOARD_TYPE_DD)
    FP32  DDSpan;
    
    OS_ENTER_CRITICAL();
    if(g_stSigCfg.ElecConst == constant)
    {
        OS_EXIT_CRITICAL();
        return;
    }
    g_stSigCfg.ElecConst = constant ;       
    OS_EXIT_CRITICAL();
    
    DD_Set_ElecConst(constant);
    DDSpan = DD_Get_Span();

    OS_ENTER_CRITICAL();
    if(g_stSigCfg.DDHighAlarm > DDSpan)
    {
        g_stSigCfg.DDLowAlarm = 0;
        g_stSigCfg.DDHighAlarm = DDSpan ;
    }
    if(g_stMainCfg.CoorFullScale[0] > DDSpan)
    {
        g_stMainCfg.CoorZeroPoint[0] = 0;
        g_stMainCfg.CoorFullScale[0] = DDSpan;  
    }
    if((g_stMainCfg.OutputSpecify[0] & 0x1) == 0){
        if(g_stMainCfg.OutputFullScale[0] > DDSpan)
        {
            g_stMainCfg.OutputZeroPoint[0] = 0;
            g_stMainCfg.OutputFullScale[0] = DDSpan; 
        }
    }
    if((g_stMainCfg.OutputSpecify[1] & 0x1) == 0){
        if(g_stMainCfg.OutputFullScale[1] > DDSpan)
        {
            g_stMainCfg.OutputZeroPoint[1] = 0;
            g_stMainCfg.OutputFullScale[1] = DDSpan; 
        }
    }     
    
    OS_EXIT_CRITICAL();
    Hart_Transmitter_SetUpper(HK338_COND,DDSpan);
    App_MeterPara_Save();
    App_SigPara_Save();
    App_Record_SaveEvent(EVENT_ElecConst);//保存事件
#endif                        
}

/*
*********************************************************************************************************
描述：   智能工厂  电导率设置电极常数
参数：   type 温度补偿方式   manaul_temp 手动时的温度
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_CondSetTempComp(INT8U type,INT16U manaul_temp)
{
    FP32  f;
    f = manaul_temp/10.f;
    
    OS_ENTER_CRITICAL();
    if((type == AUTO) && (type == g_stSigCfg.TempCompStyle))
    {
        OS_EXIT_CRITICAL();
        return;
    }
    if((type == MANUAL) && (type == g_stSigCfg.TempCompStyle) && (f == g_stSigCfg.ManualCompTemp))
    {
        OS_EXIT_CRITICAL();
        return;
    }
    g_stSigCfg.TempCompStyle = type ;          
    if(type == MANUAL)
    {
        g_stSigCfg.ManualCompTemp = manaul_temp/10.f;
    }        
    OS_EXIT_CRITICAL();
     
    App_SigPara_Save();//保存参数
    App_Record_SaveEvent(EVENT_TempComp);//保存事件
        
}

/*
*********************************************************************************************************
描述：   智能工厂  电导率设置电极常数
参数：   type 气压补偿方式   manaul_atmos 手动时的气压
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_DOSetAtmosComp(INT8U type,INT16U manaul_atmos)
{
#if(BOARD_TYPE==BOARD_TYPE_DO)
    FP32  f;
    f = manaul_atmos/10.f;
    
    OS_ENTER_CRITICAL();
    if((type == AUTO) && (type == g_stSigCfg.AtmosCompStyle))
    {
        OS_EXIT_CRITICAL();
        return;
    }
    if((type == MANUAL) && (type == g_stSigCfg.AtmosCompStyle) && (f == g_stSigCfg.ManualCompAtmos))
    {
        OS_EXIT_CRITICAL();
        return;
    }
    g_stSigCfg.AtmosCompStyle = type ;          
    if(type == MANUAL)
    {
        g_stSigCfg.ManualCompAtmos = manaul_atmos/10.f;
    }        
    OS_EXIT_CRITICAL();
     
    App_SigPara_Save();//保存参数
    App_Record_SaveEvent(EVENT_AtmosComp);//保存事件
#endif        
}


/*
*********************************************************************************************************
描述：   智能工厂  更新数据
参数：   无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_Update(void)
{
    const INT32U Pow10[10] = {
      1 , 10, 100, 1000, 10000,
      100000, 1000000, 10000000, 100000000, 1000000000
    };
    union{//2019.7.5   lyy  f[3]->f[4]
       FP32 f[4];
       INT16U i[8]; 
       INT16S si[8];  
     } RS485_Send_Data;
     
    FP32 tmp;
    OS_ENTER_CRITICAL();
     RS485_Send_Data.i[0] = SIG_COMM_FLAG_OK;
    if(g_stFlagVar.SigTimeOut == 1)
        RS485_Send_Data.i[0] |= SIG_COMM_FLAG_TIMEOUT;
    if(g_stFlagVar.SigCrcErr == 1)
        RS485_Send_Data.i[0] |= SIG_COMM_FLAG_CRCERR;
    OS_EXIT_CRITICAL(); 
    WriteModbusSlaveReg(1,RS485_Send_Data.i,SIG_COMM_FLAG);
    
#if(BOARD_TYPE==BOARD_TYPE_DD)
    OS_ENTER_CRITICAL();
    if(g_stAppVar.bUnit_mg_L == FALSE){
        RS485_Send_Data.i[0] = g_stAppVar.MeasCond*Pow10[g_stAppVar.CondShift] + 0.5;
        RS485_Send_Data.i[1] = 0;//uS/cm
    }else{
        RS485_Send_Data.i[0] = g_stAppVar.MeasCond/1000*Pow10[g_stAppVar.CondShift] + 0.5;
        RS485_Send_Data.i[1] = 1;//mS/cm
    }
    RS485_Send_Data.i[2] = g_stAppVar.CondShift;
    RS485_Send_Data.i[3] = g_stAppVar.MeasTemp * 10 + 0.5; 
    OS_EXIT_CRITICAL(); 
    WriteModbusSlaveReg(4,RS485_Send_Data.i,MEAS_REG);
#elif(BOARD_TYPE==BOARD_TYPE_DO)
    OS_ENTER_CRITICAL();
    if(ISFLOATLT(g_stAppVar.MeasConc,200,0.0001)){
        RS485_Send_Data.i[0] = g_stAppVar.MeasConc*10 + 0.5;
        RS485_Send_Data.i[1] = 0;//ug/L
    }else{
        RS485_Send_Data.i[0] = g_stAppVar.MeasConc/1000*100 + 0.5;
        RS485_Send_Data.i[1] = 1;//mg/L
    }
    
    RS485_Send_Data.i[2] = g_stAppVar.MeasTemp * 10 + 0.5; 
    RS485_Send_Data.i[3] = g_stAppVar.MeasAtmos * 10 + 0.5;  //kPa  
    RS485_Send_Data.si[4] = g_stAppVar.ElecCur;//电流整数部分 
    if(g_stAppVar.ElecCur>=0)
    {
        RS485_Send_Data.si[5] = (g_stAppVar.ElecCur-(FP32)RS485_Send_Data.si[4])*10000 + 0.5; 
    }
    else
    {
        RS485_Send_Data.si[5] = (g_stAppVar.ElecCur-(FP32)RS485_Send_Data.si[4])*10000 - 0.5; 
    }
    RS485_Send_Data.i[6] = g_stAppVar.O2Saturation*100 + 0.5;
    
    OS_EXIT_CRITICAL(); 
    WriteModbusSlaveReg(7,RS485_Send_Data.i,MEAS_REG);//2019.7.5 6->7
#elif(BOARD_TYPE==BOARD_TYPE_PH)
    OS_ENTER_CRITICAL();
    RS485_Send_Data.i[0] = g_stAppVar.MeasPH * 100 + 0.5;
    RS485_Send_Data.i[1] = g_stAppVar.MeasTemp * 10 + 0.5; 
    if(g_stAppVar.MeasVolt>0)
    {
        RS485_Send_Data.si[2] = g_stAppVar.MeasVolt * 10 + 0.5; 
    }
    else
    {
        RS485_Send_Data.si[2] = g_stAppVar.MeasVolt * 10 - 0.5; 
    }
    
    OS_EXIT_CRITICAL(); 
    WriteModbusSlaveReg(3,RS485_Send_Data.i,MEAS_REG);
    
    OS_ENTER_CRITICAL();
    if(g_stSigCfg.E0>=0)
        RS485_Send_Data.si[0] = g_stSigCfg.E0 * 10 + 0.5; 
    else
        RS485_Send_Data.si[0] = g_stSigCfg.E0 * 10 - 0.5; 
    RS485_Send_Data.i[1] = g_stSigCfg.S * 100 + 0.5; 
    OS_EXIT_CRITICAL(); 
    
    //2018.1.4  增加校准过程数据
    tmp = PH_Get_Senor_Temporary_E0();
    if(tmp>=0)
        RS485_Send_Data.si[2] = tmp*10 + 0.5;
    else
        RS485_Send_Data.si[2] = tmp*10 - 0.5;
    RS485_Send_Data.i[3] = PH_Get_Senor_Temporary_S()*100 + 0.5;
    WriteModbusSlaveReg(4,RS485_Send_Data.i,CALI_REG);
#endif
    
}

void SmartFactory_VirtualKey(INT32U key)
{
    static INT32U key_val;
    if(key!=START_CMD_KEY_ENTER && 
    key != START_CMD_KEY_ESC  &&
    key != START_CMD_KEY_CURVE1 &&
    key != START_CMD_KEY_CURVE2 &&
    key != START_CMD_KEY_UP   &&
    key != START_CMD_KEY_DOWN &&
    key != START_CMD_KEY_LEFT &&
    key != START_CMD_KEY_RIGHT)
    {
        return;
    }
    key_val = key;
    OSMboxPost(KeyMbox,(void*)&key_val);
}

/*
*********************************************************************************************************
描述：   像菜单界面发送虚拟按键 enter
参数：   无
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_VirtualKey_Enter(void)
{
    SmartFactory_VirtualKey(KEY_ENTER);
}

/*
*********************************************************************************************************
描述：   像菜单界面发送虚拟按键 设置数值 小数不可动
参数：   dest 设置目标数值
         cur  当前数值
         num  包括符号（如果有符号）和数字在内的长度，不包括小数点
         sign   NO_SIGN_MODE 表示没有符号
                SIGN_MODE  表示有符号
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_VirtualKey_Edit(INT32S dest, INT32S cur, INT8U num, INT8U sign)
{
    INT8U   dest_ac[12];
    INT8U   cur_ac[12];
    INT8U i;
    INT8U *s;

    
    if(dest == cur)
    {
        SmartFactory_VirtualKey(KEY_ENTER);
        return;
    }
     
    if(sign == SIGN_MODE)
    {
        if((dest&(1u<<31))!=(cur&(1u<<31)))
            SmartFactory_VirtualKey(KEY_DOWN);
        
        SmartFactory_VirtualKey(KEY_RIGHT);
        num--;
    }

    dest = fabs(dest);
    cur = fabs(cur);
    
    
    s = dest_ac;
    GUI_AddDecShift(dest, num, 0, &s);
    s = cur_ac;
    GUI_AddDecShift(cur, num, 0, &s);
    for(i=0;i<num;i++)
    {
        INT8S def;
        def = dest_ac[i] - cur_ac[i];
        if(def==0)
        {
            SmartFactory_VirtualKey(KEY_RIGHT);
            continue;
        }
        
        if(def>0)
        {
            while(def--)
            {
                SmartFactory_VirtualKey(KEY_UP);
            }
            SmartFactory_VirtualKey(KEY_RIGHT);
            continue;
        }
        
        if(def<0)
        {
            while(def++)
            {
                SmartFactory_VirtualKey(KEY_DOWN);
            }
            SmartFactory_VirtualKey(KEY_RIGHT);
            continue;
        }
        
        
    }


    OSTimeDly(500);
    SmartFactory_VirtualKey(KEY_ENTER);
    
    
    
}

/*
*********************************************************************************************************
描述：   智能工厂  等待数据更新
参数：   timeout 等待的系统嘀嗒  0-永久等待
返回值： 0-获得数据更新   1-超时
**********************************************************************************************************
*/
INT8U SmartFactory_Wait(INT32U timeout)
{
    INT8U err;
    OSSemPend (SemSmartFctory, timeout, &err);
    ModbusSlaveReg[ERROER_CODE] = err;
    if(err==OS_ERR_NONE)
        return 0u;
    if(err==OS_ERR_TIMEOUT)
        return 1u;
    
    return 0xFF;
}

/*
*********************************************************************************************************
描述：   智能工厂  通知收到数据
参数：   timeout 等待的系统嘀嗒  0-永久等待
返回值： 0-获得数据更新   1-超时
**********************************************************************************************************
*/
void SmartFactory_Run(void)
{
    OSSemPost (SemSmartFctory);
}



/*
*********************************************************************************************************
描述：   智能工厂  设置系统状态
参数：   state  系统状态标号
返回值： 无
**********************************************************************************************************
*/
void SmartFactory_SetSysState(INT16U sys_state)
{
    WriteModbusSlaveReg(1,&sys_state,SYSTEM_STATE);
}
/*
*********************************************************************************************************
描述：   智能工厂系统任务
参数：  parg
返回值： 
**********************************************************************************************************
*/
void TaskSmart (void  *parg)
{
    
    
    (void)parg;

	while (1) 
	{     
        if(SmartFactory_Wait(OS_TICKS_PER_SEC))//timeout
        {
            SmartFactory_Update();
        }        
        else
        {
            INT8U type;
            INT16U cmd;
            INT16U para1;
            INT16U sys_state;
            
            SmartFactory_Update();
                
                
            OS_ENTER_CRITICAL();
            cmd = REG_CMD;
            para1 = REG_PARA1;
            sys_state = ModbusSlaveReg[SYSTEM_STATE];
            OS_EXIT_CRITICAL();
            type = (cmd>>8)&0xF;
            switch(type)
            {
                case SMART_TYPE_KEY:
                    SmartFactory_VirtualKey(cmd);
                    break;
                case SMART_TYPE_RELAY:
                {
                    INT8U id = ((cmd>>4)&0xF)-1;
                    INT8U state;
                    if((cmd&0xF)==0)
                        state = 0;
                    else
                        state = 1;
                    SmartFactory_Relay(id,state);   
                                            
                    break;
                }
                case SMART_TYPE_CURRENT:
                {
                    INT16U id;
              
                    id = ((cmd>>4)&0xF)-1;
                    
                    switch(cmd)
                    {
                        case SMART_CMD_CURRENT1_CAL_START:
                        case SMART_CMD_CURRENT2_CAL_START:
                            SmartFactory_CurrentCalStart(id);
                        break;
                        case SMART_CMD_CURRENT1_CAL_END:
                        case SMART_CMD_CURRENT2_CAL_END:
                            SmartFactory_CurrentCalEnd(id);
                        break;
                        case SMART_CMD_CURRENT1_OUTPUT4MA:
                        case SMART_CMD_CURRENT2_OUTPUT4MA:
                            SmartFactory_Current(id,400);
                        break;
                        case SMART_CMD_CURRENT1_OUTPUT20MA:
                        case SMART_CMD_CURRENT2_OUTPUT20MA:
                            SmartFactory_Current(id,2000);
                        break;
                        case SMART_CMD_CURRENT1_OUTPUT12MA:
                        case SMART_CMD_CURRENT2_OUTPUT12MA:
                            SmartFactory_Current(id,1200);
                        break;
                        default:
                            break;
                                
                    }
                    
                    break;
                }
                    
                case SMART_TYPE_BEEP:
                    SmartFactory_Beep(cmd&0x1);
                    break;
                case SMART_TYPE_MAIN:
                    if(cmd==SMART_CMD_VIEW_RUNLOG)
                    {
                        SmartFactory_VeiwRunLog();
                    }
                    if(cmd==SMART_CMD_VIEW_MAIN)
                    {
                        App_MeterPara_Get();
                        OS_ENTER_CRITICAL();
                        FuncLevel = FID_Measure;
                        g_stFlagVar.SysState = DEL_MENU_STATE; 
                        OS_EXIT_CRITICAL();
                    }
                    if(cmd==SMART_CMD_VIEW_MENU)
                    {
                        App_MeterPara_Get();
                        OS_ENTER_CRITICAL();
                        FuncLevel = FID_MenuMain;
                        g_stFlagVar.SysState = DEL_MENU_STATE; 
                        OS_EXIT_CRITICAL();
                    }
                    if(cmd==SMART_CMD_SET_TIME)
                    {
                        if(gFM24CL04_COM == 0xA0)
                        {
                            RTCDate    Date;
                            RTCTime    Time;
                            OS_ENTER_CRITICAL();
                            Date.RTC_Year = REG_PARA1;
                            Date.RTC_Mon = REG_PARA2;
                            Date.RTC_Mday = REG_PARA3;
                            Time.RTC_Hour = REG_PARA4;
                            Time.RTC_Min = REG_PARA5;
                            Time.RTC_Sec = REG_PARA6;
                            OS_EXIT_CRITICAL();
                            Lpc17xx_RTC_SetTime(&Time);
                            Lpc17xx_RTC_SetDate(&Date);
                        }
                        else
                        {
                            OS_ENTER_CRITICAL();
                            g_stDateTime.Year = (REG_PARA1)%100;//2019.7.5
                            g_stDateTime.Mon = REG_PARA2;
                            g_stDateTime.Mday = REG_PARA3;
                            g_stDateTime.Hour = REG_PARA4;
                            g_stDateTime.Min = REG_PARA5;
                            g_stDateTime.Sec = REG_PARA6;
                            OS_EXIT_CRITICAL();
                            P8563_Set_Time();
                        }
                    }
                    if(cmd==SMART_CMD_CHECK_FLASH)
                    {
                        LOG_REC	RunRec;
                        INT16U flash_state = 0xFFFF;
                        //铁电
                        OS_ENTER_CRITICAL();
#if(BOARD_TYPE==BOARD_TYPE_DD)
                        if(g_stRecHeader[REC_RUN].Num<3)
                        {
                            flash_state &= ~0x1;
                        }
#else
                        if(g_stMainCfg.LogRunNum<3)
                        {
                            flash_state &= ~0x1;
                        }
#endif
                        
                        //时间  2019.7.5
                        if(gFM24CL04_COM == 0xA0)
                        {
                            if(g_stDateTime.Year<2018)
                            {
                                flash_state &= ~0x4;
                            }
                        }
                        else
                        {
                            if(g_stDateTime.Year<18)
                            {
                                flash_state &= ~0x4;
                            }
                        }
                        
                        OS_EXIT_CRITICAL();
                        //AT45
#if(BOARD_TYPE==BOARD_TYPE_DD)
                        RecordRead(&g_stRecHeader[REC_RUN],MAX_RUNREC_NUM-3,1,(INT8U *)&RunRec);
#else
                        App_Record_Read(&g_stRunRec,MAX_RUNREC_NUM-3,1,(INT8U *)&RunRec);
#endif
                        if(RunRec.Minute>59)
                        {
                            flash_state &= ~0x2;
                        }
                        WriteModbusSlaveReg(1,&flash_state,FLASH_STATE);
                    }
                    //2018.11.6
                    if(cmd==SMART_CMD_SET_485ADDR)
                    {
                        if((REG_PARA1>=1)&&(REG_PARA1<=254))
                        {
                            OS_ENTER_CRITICAL();
                            g_stMainCfg.CommLocalAddress = (INT8U)REG_PARA1 ;             
                            OS_EXIT_CRITICAL();
                            App_MeterPara_Save();//保存参数
                            App_Record_SaveEvent(EVENT_CommLocalAddress);//保存事件
                            changeToSYSMode();
                            Modbus_Slave_Init(g_stMainCfg.CommLocalAddress,NULL,NULL);
                            changeToUSRMode();
                        }
                    
                    }
                    break;
#if(BOARD_TYPE==BOARD_TYPE_DO)
                case SMART_TYPE_DO:
                    if(cmd==SMART_CMD_DO_ELEC_CAL_START)
                    {
                        SmartFactory_DOElecCal();
                    }
                    if(cmd==SMART_CMD_DO_AIR_CAL_START)
                    {
                        SmartFactory_DOAirCal();
                    }
                    if(cmd==SMART_CMD_DO_ZERO_CAL_START)
                    {
                        SmartFactory_DOZeroCal();
                    }
                    if(cmd==SMART_CMD_DO_ATMOS_CAL_START)
                    {
                        SmartFactory_DOAtmosCal(para1);
                    }
                    break;
#endif
#if(BOARD_TYPE==BOARD_TYPE_PH)
                case SMART_TYPE_PH:
                    if(cmd==SMART_CMD_PH_ELEC_CAL_START)
                    {
                        SmartFactory_PHElecCal();
                    } 
                    else if(sys_state ==SYS_STATE_CAL_PH_ELEC_INPUT1  && cmd==SMART_CMD_PH_ELEC_CAL_N500MV)
                    {
                        SmartFactory_VirtualKey_Edit((INT16S)para1,-5000,6,SIGN_MODE);
                    }
                    else if(sys_state ==SYS_STATE_CAL_PH_ELEC_INPUT2  && cmd==SMART_CMD_PH_ELEC_CAL_P500MV)
                    {
                        SmartFactory_VirtualKey_Edit((INT16S)para1,+5000,6,SIGN_MODE);
                    }
                    else if(cmd==SMART_CMD_PH_BUFFER_CAL_START)
                    {
                        SmartFactory_PHBufferCal();
                    }
                    
                    break;
#endif
#if(BOARD_TYPE==BOARD_TYPE_DD)
                case SMART_TYPE_COND:
                    if(cmd==SMART_CMD_COND_ELEC_CAL_START)
                    {
                        SmartFactory_CondElecCal();
                    }
                    else if(cmd>SMART_CMD_COND_ELEC_CAL_START&&cmd<=SMART_CMD_COND_ELEC_CAL_50OHMS)
                    {
                        if(sys_state == SYS_STATE_CAL_COND_ELEC_INPUTRESTIPS)
                        {
                            SmartFactory_VirtualKey_Enter();
                        }
                    }
                    else if(cmd==SMART_CMD_COND_SET_CELL_CONSTANT)
                    {
                        SmartFactory_CondSetCellConstant(para1/100.0f);
                    }
                    else if(cmd==SMART_CMD_COND_RANGE_AND_FREQ_AUTO)
                    {
                        if(REG_PARA1==0 || REG_PARA1>4)
                        {
                            OS_ENTER_CRITICAL();
                            g_stAppVar.ElecTestEn = CLOSE;
                            OS_EXIT_CRITICAL();
                            DD_ManuSet_RangSetEnd();
                        }
                        else
                        {
                            OS_ENTER_CRITICAL();
                            g_stAppVar.ElecTestEn = OPEN;
                            g_stAppVar.ElecTestRange = REG_PARA1;//1~4
                            if(REG_PARA2 == 300)
                                g_stAppVar.ElecTestFreq = 0;
                            else
                                g_stAppVar.ElecTestFreq = 1;
                            OS_EXIT_CRITICAL();
                            DD_ManuSet_RangSetBegin(REG_PARA1,REG_PARA2);
                        }
                            
                    }
                    break;
#endif
                case SMART_TYPE_TEMP:
                    if(cmd==SMART_CMD_TEMP_ELEC_CAL_TEMP1)
                    {
                        SmartFactory_TempElecCal();
                    }
                    else if(cmd==SMART_CMD_TEMP_ELEC_CAL_TEMP2 && sys_state == SYS_STATE_CAL_TEMP_ELEC_FIRSTTIPS2) 
                    {
                        SmartFactory_VirtualKey_Enter();
                    }
                    else if(cmd==SMART_CMD_COND_SET_TEMP_COMP)
                    {
                        SmartFactory_CondSetTempComp(AUTO,0);
                    }
                    else if(cmd==SMART_CMD_TEMP_COMP_MANUAL_25)
                    {
                        //SmartFactory_CondSetTempComp(MANUAL,250);
                        //2018.9.13  lyy
                        SmartFactory_CondSetTempComp(MANUAL,REG_PARA1);
                    }
                    else if(cmd==SMART_CMD_ATMOS_COMP_AUTO)
                    {
                        SmartFactory_DOSetAtmosComp(AUTO,0);
                    }
                    else if(cmd==SMART_CMD_ATMOS_COMP_MANUAL_1013)
                    {
                        SmartFactory_DOSetAtmosComp(MANUAL,1013);
                    }
                    break;
                default:
                    break;
                
            }
        }
        
    }
}

