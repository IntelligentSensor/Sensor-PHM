## Intelligent Ph Overview

致力于智能PH项目的高效交流与协同开发.

在HK-328 pH4.0软、硬件基础上，升级软件实现故障诊断和生命周期管理（Prognostic and Health Management），以提高仪表维护管理水平，实现仪表过程数据化和记录文档化。

## 仪表数据及存储说明

### 数据分类
- 原始数据
时间、实时PH、实时温度、电极编号；
- 过程数据
斜率、零点、放入缓冲液响应时间、放入样品液响应时间；校准、维护计数；
- 结果数据
数据形态（正常、一次跳跃、多次跳跃、数值波动、凸奇异点、凹奇异点、单调连续上升或下降、固定值、负值）；数值波动故障、数值跳跃故障、溶液地未连接故障、流通池接地故障；环境压力指数、电极剩余寿命、下次校准时间、下次维护时间。
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

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/intelligentph/PhRepository/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
