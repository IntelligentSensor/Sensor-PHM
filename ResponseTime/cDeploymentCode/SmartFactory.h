/********************************************************************
*文件名:SmartFactory.h
*修改日期:2017.9.7
*备注:

2018-01-04
*********************************************************************/

#ifndef __SMARTFACTORY_H
#define __SMARTFACTORY_H

/*********************************************************
系统预留64个寄存器    读命令3  写命令16
0~31    测量变量    
32~48   系统状态变量
48~63   控制命令变量
*********************************************************/
//READ  MEAS   0
#define MEAS_REG 0
#define CALI_REG 16
#define ERROER_CODE  31
//READ  SYS  STATE   32
#define MAIN_BOARD_TYPE 32
#define MAIN_BOARD_VER  33
#define SIG_BOARD_VER   34
#define SYSTEM_STATE    35
#define SIG_COMM_FLAG   36
#define FLASH_STATE     37

//WRITE-ONLY   48
extern INT16U ModbusSlaveReg[];
#define REG_CMD ModbusSlaveReg[48]
#define REG_PARA1 ModbusSlaveReg[49]
#define REG_PARA2 ModbusSlaveReg[50]
#define REG_PARA3 ModbusSlaveReg[51]
#define REG_PARA4 ModbusSlaveReg[52]
#define REG_PARA5 ModbusSlaveReg[53]
#define REG_PARA6 ModbusSlaveReg[55]

#define SIG_COMM_FLAG_OK 0
#define SIG_COMM_FLAG_TIMEOUT 1
#define SIG_COMM_FLAG_CRCERR 2


//溶解氧
#define SDO_CALI_TMP_X10V_REG  (CALI_REG)
#define SDO_CALI_TMP_X1V_REG   (CALI_REG+1)
#define SDO_CALI_TMP_A_REG  (CALI_REG+2)
#define SDO_CALI_TMP_ZERO_REG   (CALI_REG+3)

//定义较低优先级    注意不要和其他冲突
#define  TASK_SMART_ID                15	
#define  TASK_SMART_PRIO              TASK_SMART_ID 
#define  TASK_SMART_STK_SIZE          256
extern OS_STK  StkSmart[TASK_SMART_STK_SIZE];   
extern void TaskSmart (void  *parg);
/**************************************************************
    TYPE DEFINE
**************************************************************/
#define SMART_TYPE_KEY                          0    
#define SMART_TYPE_RELAY                        1
#define SMART_TYPE_CURRENT                      2
#define SMART_TYPE_BEEP                         3
#define SMART_TYPE_MAIN                         4
#define SMART_TYPE_DO                           5
#define SMART_TYPE_PH                           6
#define SMART_TYPE_COND                         7
#define SMART_TYPE_TEMP                         8

/**************************************************************
    CMD DEFINE
**************************************************************/
#define START_CMD_KEY_ENTER                     KEY_ENTER
#define START_CMD_KEY_ESC                       KEY_ESC
#define START_CMD_KEY_CURVE1                    KEY_CURVE1
#define START_CMD_KEY_CURVE2                    KEY_CURVE2
#define START_CMD_KEY_UP                        KEY_UP
#define START_CMD_KEY_DOWN                      KEY_DOWN
#define START_CMD_KEY_LEFT                      KEY_LEFT
#define START_CMD_KEY_RIGHT                     KEY_RIGHT

#define SMART_CMD_RELAY1_OPEN                   0x110	//300电源板继电器1断
#define SMART_CMD_RELAY1_CLOSE                  0x111	//300电源板继电器1通
#define SMART_CMD_RELAY2_OPEN                   0x120	//300电源板继电器2断
#define SMART_CMD_RELAY2_CLOSE                  0x121	//300电源板继电器2通
#define SMART_CMD_CURRENT1_CAL_START            0x210	//300主板电流1进入20mA校准菜单
#define SMART_CMD_CURRENT1_CAL_END              0x211	//300主板电流1校准完毕保存校准值
#define SMART_CMD_CURRENT1_OUTPUT4MA            0x212	//300主板电流1输出4mA电流
#define SMART_CMD_CURRENT1_OUTPUT20MA           0x213	//300主板电流1输出20mA电流
#define SMART_CMD_CURRENT1_OUTPUT12MA           0x214	//300主板电流1输出12mA电流
#define SMART_CMD_CURRENT2_CAL_START            0x220	//300主板电流2进入20mA校准菜单
#define SMART_CMD_CURRENT2_CAL_END              0x221	//300主板电流2校准完毕保存校准值
#define SMART_CMD_CURRENT2_OUTPUT4MA            0x222	//300主板电流2输出4mA电流
#define SMART_CMD_CURRENT2_OUTPUT20MA           0x223	//300主板电流2输出20mA电流
#define SMART_CMD_CURRENT2_OUTPUT12MA           0x224	//300主板电流2输出12mA电流
#define SMART_CMD_BEEP_OPEN                     0x300	//300主板关闭蜂鸣器
#define SMART_CMD_BEEP_CLOSE                    0x301	//300主板打开蜂鸣器
#define SMART_CMD_VIEW_RUNLOG                   0x401	//300主板进入运行记录菜单
#define SMART_CMD_VIEW_MAIN                     0x402   //300主板返回主界面
#define SMART_CMD_VIEW_MENU                     0x403   //300主板进入主菜单
#define SMART_CMD_SET_TIME                      0x404   //300主板设置时间和日期
#define SMART_CMD_CHECK_FLASH                   0x405   //检查铁电 AT45 时间   ,必须保证已经断电重启
#define SMART_CMD_SET_485ADDR                   0x406   //更改485地址//2018.10.6
#define SMART_CMD_DO_ELEC_CAL_START             0x501	//318主板进入电气校准菜单（溶解氧）
#define SMART_CMD_DO_AIR_CAL_START              0x502	//318主板进入空气校准菜单（溶解氧）
#define SMART_CMD_DO_ZERO_CAL_START             0x503	//318主板进入零点校准菜单（溶解氧）
#define SMART_CMD_DO_ATMOS_CAL_START            0x504   //318进入气压校准
#define SMART_CMD_PH_ELEC_CAL_START             0x601	//328主板进入电气校准菜单（PH）
#define SMART_CMD_PH_ELEC_CAL_N500MV            0x602	//328主板进入-500mV值校准
#define SMART_CMD_PH_ELEC_CAL_P500MV            0x603	//328主板进入500mV值校准
#define SMART_CMD_PH_BUFFER_CAL_START           0x611   //328进入pH缓冲液校准
#define SMART_CMD_COND_SET_CELL_CONSTANT        0x701	//338主板进入电极常数设置为0.01
#define SMART_CMD_COND_ELEC_CAL_START           0x703   //338电气校准--无穷大
#define SMART_CMD_COND_ELEC_CAL_50KOHMS         0x704   //338电气校准--50K电阻接完毕
#define SMART_CMD_COND_ELEC_CAL_5KOHMS          0x705   //338电气校准--5K电阻接完毕
#define SMART_CMD_COND_ELEC_CAL_500OHMS         0x706   //338电气校准--500电阻接完毕
#define SMART_CMD_COND_ELEC_CAL_50OHMS          0x707   //338电气校准--50电阻接完毕
#define SMART_CMD_COND_RANGE_AND_FREQ_AUTO      0x708   //338手动输入档位和激励开启参数部分分别是档位和激励
#define SMART_CMD_COND_SET_TEMP_COMP            0x801	//338主板设置温度补偿方式为自动
#define SMART_CMD_TEMP_ELEC_CAL_TEMP1           0x802   //温度电气校准第一点接完毕
#define SMART_CMD_TEMP_ELEC_CAL_TEMP2           0x803   //温度电气校准第二点接完毕
#define SMART_CMD_TEMP_COMP_AUTO                0x804   //自动温度补偿
#define SMART_CMD_TEMP_COMP_MANUAL_25           0x805   //手动温度补偿
#define SMART_CMD_ATMOS_COMP_AUTO               0x806   //自动气压补偿
#define SMART_CMD_ATMOS_COMP_MANUAL_1013        0x807   //手动气压补偿
/*********************************************************************
系统状态宏定义
***********************************************************************/
#define SYS_STATE_INIT                              0U

#define SYS_STATE_CAL_OUTPUT1_BASE                  0x210u
#define SYS_STATE_CAL_OUTPUT1_CAL                   (SYS_STATE_CAL_OUTPUT1_BASE+0x0u)
#define SYS_STATE_CAL_OUTPUT1_CALOK                 (SYS_STATE_CAL_OUTPUT1_BASE+0x1u)
#define SYS_STATE_CAL_OUTPUT1_CALLOW                (SYS_STATE_CAL_OUTPUT1_BASE+0x2u)
#define SYS_STATE_CAL_OUTPUT1_CALHIGH               (SYS_STATE_CAL_OUTPUT1_BASE+0x3u)
#define SYS_STATE_CAL_OUTPUT1_RESTART               (SYS_STATE_CAL_OUTPUT1_BASE+0x4u)
#define SYS_STATE_CAL_OUTPUT1_ESC                   (SYS_STATE_CAL_OUTPUT1_BASE+0x5u)

#define SYS_STATE_CAL_OUTPUT2_BASE                  0x220u
#define SYS_STATE_CAL_OUTPUT2_CAL                   (SYS_STATE_CAL_OUTPUT2_BASE+0x0u)
#define SYS_STATE_CAL_OUTPUT2_CALOK                 (SYS_STATE_CAL_OUTPUT2_BASE+0x1u)
#define SYS_STATE_CAL_OUTPUT2_CALLOW                (SYS_STATE_CAL_OUTPUT2_BASE+0x2u)
#define SYS_STATE_CAL_OUTPUT2_CALHIGH               (SYS_STATE_CAL_OUTPUT2_BASE+0x3u)
#define SYS_STATE_CAL_OUTPUT2_RESTART               (SYS_STATE_CAL_OUTPUT2_BASE+0x4u)
#define SYS_STATE_CAL_OUTPUT2_ESC                   (SYS_STATE_CAL_OUTPUT2_BASE+0x5u)

#define SYS_STATE_CAL_DO_ELEC_BASE                  0x500u
#define SYS_STATE_CAL_DO_ELEC_FIRSTTIPS1            (SYS_STATE_CAL_DO_ELEC_BASE+0x0u)
#define SYS_STATE_CAL_DO_ELEC_CAL1                  (SYS_STATE_CAL_DO_ELEC_BASE+0x1u)
#define SYS_STATE_CAL_DO_ELEC_CAL2                  (SYS_STATE_CAL_DO_ELEC_BASE+0x2u)
#define SYS_STATE_CAL_DO_ELEC_CALOK                 (SYS_STATE_CAL_DO_ELEC_BASE+0x3u)
#define SYS_STATE_CAL_DO_ELEC_CALLOW                (SYS_STATE_CAL_DO_ELEC_BASE+0x4u)
#define SYS_STATE_CAL_DO_ELEC_CALHIGH               (SYS_STATE_CAL_DO_ELEC_BASE+0x5u)
#define SYS_STATE_CAL_DO_ELEC_RESTART               (SYS_STATE_CAL_DO_ELEC_BASE+0x6u)
#define SYS_STATE_CAL_DO_ELEC_ESC                   (SYS_STATE_CAL_DO_ELEC_BASE+0x7u)

#define SYS_STATE_CAL_DO_AIR_BASE                   0x510u
#define SYS_STATE_CAL_DO_AIR_FIRSTTIPS              (SYS_STATE_CAL_DO_AIR_BASE+0x0u)
#define SYS_STATE_CAL_DO_AIR_CAL                    (SYS_STATE_CAL_DO_AIR_BASE+0x1u)
#define SYS_STATE_CAL_DO_AIR_CalOK                  (SYS_STATE_CAL_DO_AIR_BASE+0x2u)
#define SYS_STATE_CAL_DO_AIR_CALLOW                 (SYS_STATE_CAL_DO_AIR_BASE+0x3u)
#define SYS_STATE_CAL_DO_AIR_CALHIGH                (SYS_STATE_CAL_DO_AIR_BASE+0x4u)
#define SYS_STATE_CAL_DO_AIR_RESTART                (SYS_STATE_CAL_DO_AIR_BASE+0x5u)
#define SYS_STATE_CAL_DO_AIR_ESC                    (SYS_STATE_CAL_DO_AIR_BASE+0x6u)

#define SYS_STATE_CAL_DO_ZERO_BASE                  0x520u
#define SYS_STATE_CAL_DO_ZERO_FIRSTTIPS             (SYS_STATE_CAL_DO_ZERO_BASE+0x0u) 
#define SYS_STATE_CAL_DO_ZERO_CAL                   (SYS_STATE_CAL_DO_ZERO_BASE+0x1u) 
#define SYS_STATE_CAL_DO_ZERO_CALOK                 (SYS_STATE_CAL_DO_ZERO_BASE+0x2u) 
#define SYS_STATE_CAL_DO_ZERO_CALLOW                (SYS_STATE_CAL_DO_ZERO_BASE+0x3u) 
#define SYS_STATE_CAL_DO_ZERO_CALHIGH               (SYS_STATE_CAL_DO_ZERO_BASE+0x4u) 
#define SYS_STATE_CAL_DO_ZERO_RESTART               (SYS_STATE_CAL_DO_ZERO_BASE+0x5u) 
#define SYS_STATE_CAL_DO_ZERO_ESC                   (SYS_STATE_CAL_DO_ZERO_BASE+0x6u) 

#define SYS_STATE_CAL_DO_ATMOS_BASE                 0x530u
#define SYS_STATE_CAL_DO_ATMOS_FIRSTTIPS            (SYS_STATE_CAL_DO_ATMOS_BASE+0x0u) 
#define SYS_STATE_CAL_DO_ATMOS_CAL                  (SYS_STATE_CAL_DO_ATMOS_BASE+0x1u) 
#define SYS_STATE_CAL_DO_ATMOS_CALOK                (SYS_STATE_CAL_DO_ATMOS_BASE+0x2u) 
#define SYS_STATE_CAL_DO_ATMOS_CALLOW               (SYS_STATE_CAL_DO_ATMOS_BASE+0x3u) 
#define SYS_STATE_CAL_DO_ATMOS_CALHIGH              (SYS_STATE_CAL_DO_ATMOS_BASE+0x4u) 
#define SYS_STATE_CAL_DO_ATMOS_RESTART              (SYS_STATE_CAL_DO_ATMOS_BASE+0x5u) 
#define SYS_STATE_CAL_DO_ATMOS_ESC                  (SYS_STATE_CAL_DO_ATMOS_BASE+0x6u) 

#define SYS_STATE_CAL_PH_ELEC_BASE                  0x600u
#define SYS_STATE_CAL_PH_ELEC_FIRSTTIPS1            (SYS_STATE_CAL_PH_ELEC_BASE+0x0u)
#define SYS_STATE_CAL_PH_ELEC_FIRSTTIPS2            (SYS_STATE_CAL_PH_ELEC_BASE+0x1u)
#define SYS_STATE_CAL_PH_ELEC_CAL1                  (SYS_STATE_CAL_PH_ELEC_BASE+0x2u)
#define SYS_STATE_CAL_PH_ELEC_CAL2                  (SYS_STATE_CAL_PH_ELEC_BASE+0x3u)
#define SYS_STATE_CAL_PH_ELEC_INPUT1                (SYS_STATE_CAL_PH_ELEC_BASE+0x4u)
#define SYS_STATE_CAL_PH_ELEC_INPUT2                (SYS_STATE_CAL_PH_ELEC_BASE+0x5u)
#define SYS_STATE_CAL_PH_ELEC_CALOK                 (SYS_STATE_CAL_PH_ELEC_BASE+0x6u)
#define SYS_STATE_CAL_PH_ELEC_SPANCALLOW            (SYS_STATE_CAL_PH_ELEC_BASE+0x7u)
#define SYS_STATE_CAL_PH_ELEC_SPANCALHIGH           (SYS_STATE_CAL_PH_ELEC_BASE+0x8u)
#define SYS_STATE_CAL_PH_ELEC_ZEROCALLOW            (SYS_STATE_CAL_PH_ELEC_BASE+0x9u)
#define SYS_STATE_CAL_PH_ELEC_ZEROCALHIGH           (SYS_STATE_CAL_PH_ELEC_BASE+0xAu)
#define SYS_STATE_CAL_PH_ELEC_RESTART               (SYS_STATE_CAL_PH_ELEC_BASE+0xBu)
#define SYS_STATE_CAL_PH_ELEC_ESC                   (SYS_STATE_CAL_PH_ELEC_BASE+0xCu)

#define SYS_STATE_CAL_PH_BUFFER_BASE                0x610u
#define SYS_STATE_CAL_PH_BUFFER_FIRSTTIPS           (SYS_STATE_CAL_PH_BUFFER_BASE+0x0u)
#define SYS_STATE_CAL_PH_BUFFER_IDENTIFYTIPS        (SYS_STATE_CAL_PH_BUFFER_BASE+0x1u)
#define SYS_STATE_CAL_PH_BUFFER_SELETTIPS           (SYS_STATE_CAL_PH_BUFFER_BASE+0x2u)
#define SYS_STATE_CAL_PH_BUFFER_CAL                 (SYS_STATE_CAL_PH_BUFFER_BASE+0x3u)
#define SYS_STATE_CAL_PH_BUFFER_CALOK               (SYS_STATE_CAL_PH_BUFFER_BASE+0x4u)
#define SYS_STATE_CAL_PH_BUFFER_CALE0LOW            (SYS_STATE_CAL_PH_BUFFER_BASE+0x5u)
#define SYS_STATE_CAL_PH_BUFFER_CALE0HIGH           (SYS_STATE_CAL_PH_BUFFER_BASE+0x6u)
#define SYS_STATE_CAL_PH_BUFFER_CALSLOW             (SYS_STATE_CAL_PH_BUFFER_BASE+0x7u)
#define SYS_STATE_CAL_PH_BUFFER_CALSHIGH            (SYS_STATE_CAL_PH_BUFFER_BASE+0x8u)
#define SYS_STATE_CAL_PH_BUFFER_RESTART             (SYS_STATE_CAL_PH_BUFFER_BASE+0x9u)
#define SYS_STATE_CAL_PH_BUFFER_ESC                 (SYS_STATE_CAL_PH_BUFFER_BASE+0xAu)

#define SYS_STATE_CAL_COND_ELEC_BASE                0x700u
#define SYS_STATE_CAL_COND_ELEC_FIRSTTIPS           (SYS_STATE_CAL_COND_ELEC_BASE+0x0u)
#define SYS_STATE_CAL_COND_ELEC_INPUTRESTIPS        (SYS_STATE_CAL_COND_ELEC_BASE+0x1u)
#define SYS_STATE_CAL_COND_ELEC_CAL                 (SYS_STATE_CAL_COND_ELEC_BASE+0x2u)
#define SYS_STATE_CAL_COND_ELEC_CALOK               (SYS_STATE_CAL_COND_ELEC_BASE+0x3u)
#define SYS_STATE_CAL_COND_ELEC_CALFAILURE          (SYS_STATE_CAL_COND_ELEC_BASE+0x4u)
#define SYS_STATE_CAL_COND_ELEC_RESTART             (SYS_STATE_CAL_COND_ELEC_BASE+0x5u)
#define SYS_STATE_CAL_COND_ELEC_ESC                 (SYS_STATE_CAL_COND_ELEC_BASE+0x6u) 

#define SYS_STATE_CAL_TEMP_ELEC_BASE                0x800u
#define SYS_STATE_CAL_TEMP_ELEC_FIRSTTIPS1          (SYS_STATE_CAL_TEMP_ELEC_BASE+0x0u)
#define SYS_STATE_CAL_TEMP_ELEC_FIRSTTIPS2          (SYS_STATE_CAL_TEMP_ELEC_BASE+0x1u)
#define SYS_STATE_CAL_TEMP_ELEC_CAL1                (SYS_STATE_CAL_TEMP_ELEC_BASE+0x2u)
#define SYS_STATE_CAL_TEMP_ELEC_CAL2                (SYS_STATE_CAL_TEMP_ELEC_BASE+0x3u)
#define SYS_STATE_CAL_TEMP_ELEC_CALOK               (SYS_STATE_CAL_TEMP_ELEC_BASE+0x4u)
#define SYS_STATE_CAL_TEMP_ELEC_CALERR              (SYS_STATE_CAL_TEMP_ELEC_BASE+0x5u)
#define SYS_STATE_CAL_TEMP_ELEC_CALSLOPELOW         (SYS_STATE_CAL_TEMP_ELEC_BASE+0x6u)
#define SYS_STATE_CAL_TEMP_ELEC_CALSLOPEHIGH        (SYS_STATE_CAL_TEMP_ELEC_BASE+0x7u)
#define SYS_STATE_CAL_TEMP_ELEC_CALZEROLOW          (SYS_STATE_CAL_TEMP_ELEC_BASE+0x8u)
#define SYS_STATE_CAL_TEMP_ELEC_CALZEROHIGH         (SYS_STATE_CAL_TEMP_ELEC_BASE+0x9u)
#define SYS_STATE_CAL_TEMP_ELEC_RESTART             (SYS_STATE_CAL_TEMP_ELEC_BASE+0xAu)
#define SYS_STATE_CAL_TEMP_ELEC_ESC                 (SYS_STATE_CAL_TEMP_ELEC_BASE+0xBu)



/*********************************************************************
函数声明
***********************************************************************/


void SmartFactory_Init(void);
void SmartFactory_Relay(INT8U id, INT8U state);
void SmartFactory_Current(INT8U id, INT16U value);
void SmartFactory_Beep(INT8U state);
void SmartFactory_ViewMain(void);
void SmartFactory_ViewMenu(void);
void SmartFactory_CurrentCalStart(INT8U id);
void SmartFactory_CurrentCalEnd(INT8U id);
void SmartFactory_VeiwRunLog(void);
void SmartFactory_DOElecCal(void);
void SmartFactory_DOAirCal(void);
void SmartFactory_DOZeroCal(void);
void SmartFactory_DOElecCal_x10Volt(FP32 volt);
void SmartFactory_DOElecCal_x1Volt(FP32 volt);
void SmartFactory_DOAirCal_a(FP32 a);
void SmartFactory_DOZeroCal_zero(FP32 zero);
void SmartFactory_PHElecCal(void);
void SmartFactory_PHBufferCal(void);
void SmartFactory_CondElecCal(void);
void SmartFactory_CondSetCellConstant(FP32 constant);
void SmartFactory_CondSetTempComp(INT8U type,INT16U manaul_temp);
void SmartFactory_Update(void);
void SmartFactory_VirtualKey(INT32U key);
void SmartFactory_VirtualKey_Enter(void);
void SmartFactory_VirtualKey_Edit(INT32S dest, INT32S cur, INT8U num, INT8U sign);
INT8U SmartFactory_Wait(INT32U timeout);
void SmartFactory_Run(void);
void SmartFactory_SetSysState(INT16U sys_state);
void TaskSmart (void  *parg);



#endif

