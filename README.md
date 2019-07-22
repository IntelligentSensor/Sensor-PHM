## Intelligent Ph Overview

致力于智能PH项目的高效交流与协同开发.

在HK-328 pH4.0软、硬件基础上，数据建模实现故障诊断和生命周期管理（Prognostic and Health Management），以提高仪表维护管理水平，实现仪表过程数据化和记录文档化。

## 仪表数据及存储说明

### 数据分类

   - 原始数据：时间、实时PH、实时温度、电极编号；
   - 过程数据：斜率、零点、放入缓冲液响应时间、放入样品液响应时间；校准、维护计数；
   - 结果数据：数据形态（数据正常、一次跳跃、多次跳跃、波动、凸奇异点、凹奇异点、单调连续上升或下降、固定值、负值）；故障（数值波动障、数值跳跃、溶液地未连接、流通池接地、采集板、温度电极、校准异常）；环境压力指数、电极剩余寿命、下次校准时间、下次维护时间。
   
### 数据格式及字段说明

   - 电极编号：electNum int 4字节   
   - 时间：Time   struct 年、月、日、时、分、秒  7字节
   - PH值：Ph   float  4字节
   - 温度：Temp  float  4字节
   - 故障类型：faultType char  CailFault、tempElect、fluctuate、Jump； 6字节
   - 数据形态：dataForm char  Normal、Jump1、Jumpn、Surge、singularityCave、singularityVex、monotone、fixed、negative； 6字节
   - 环境压力指数：ESI  float   
   - 斜率：S float  4字节   单位PH/mV
   - 零点：E0 float  4字节  单位mV
   - 缓冲液响应时间：Set1  int    4字节
   - 样品液响应时间：Set2  int    4字节
   - 校准计数：Calinum    int     4字节
   - 维护计数：Mainnum   int     4字节
   - 电极剩余寿命：Rul   float      4字节
   - 下次校准时间：Calitime  float  4字节
   - 下次维护时间：Maintime  float  4字节

## 数据分析模型实现

### 响应时间提取
1. For python verify code see [python响应时间](https://github.com/intelligentph/PhRepository/tree/master/ResponseTime/pythonVerifyCode).
which file is saved in the `reg.py` file.
2. For C Deployment code see [c响应时间](https://github.com/intelligentph/PhRepository/tree/master/ResponseTime/cDeploymentCode).
which file is saved in the `BitmapDialog.cpp` file.

### 生命周期管理
1. For python verify code see [python生命周期](https://github.com/intelligentph/PhRepository/tree/master/HealthManagement/pythonVerifyCode).
which file is saved in the `reg.py` file.
2. For C Deployment code see [c生命周期](https://github.com/intelligentph/PhRepository/tree/master/HealthManagement/cDeploymentCode).
which file is saved in the `BitmapDialog.cpp` file.

### 故障诊断与数据形态识别
1. For python verify code see [python故障](https://github.com/intelligentph/PhRepository/tree/master/Prognostic/pythonVerifyCode).
which file is saved in the `reg.py` file.
2. For C Deployment code see [c故障](https://github.com/intelligentph/PhRepository/tree/master/Prognostic/cDeploymentCode).
which file is saved in the `BitmapDialog.cpp` file.

### 环境压力评估
1. For python verify code see [python环境评估](https://github.com/intelligentph/PhRepository/tree/master/EnvironmentalStressIndex/pythonVerifyCode).
which file is saved in the `reg.py` file.
2. For C Deployment code see [c环境评估](https://github.com/intelligentph/PhRepository/tree/master/EnvironmentalStressIndex/cDeploymentCode).
which file is saved in the `BitmapDialog.cpp` file.

## 文档 notes

## 其它 Miscellaneous

![Image](src)

You can use the [Editor on github](https://github.com/intelligentph/PhRepository/edit/gh-pages/README.md) to maintain and preview the content for your website in markdown files.
