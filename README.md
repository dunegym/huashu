# 2025华数杯C题代码仓

## AI提出的大致框架

### 基于Grok的deep research

#### 关键要点
- 研究表明，LED光源可以通过调节光谱特性来优化颜色质量和健康节律效应，但具体建模方法需根据问题复杂性调整。
- 建模流程包括数据分析、参数计算、优化设计和统计分析，适合分步骤解决。
- 证据倾向于使用标准工具如LuxPy进行光谱参数计算，但优化和统计分析可能涉及多种方法。

---

#### 建模大致思路和流程框架

##### 概述
本课题涉及LED光源的可调控生物节律研究，分为四个子问题：计算光谱参数、设计多通道LED系统、模拟日光变化和分析睡眠数据。以下是针对每个子问题的建模思路和流程框架，旨在为用户提供清晰的解决路径。

##### 问题1：计算光谱参数
- **目标**：从给定的光谱功率分布（SPD）数据计算相关色温（CCT）、距离普朗克轨迹的距离（Duv）、保真度指数（Rf）、色域指数（Rg）和褪黑素日光照度比（mel-DER）。
- **方法**：
  - 使用标准方法将SPD转换为CIE XYZ三刺激值，然后计算CCT和Duv。
  - 采用IES TM-30-20方法计算Rf和Rg，涉及测试颜色集的颜色差异计算。
  - 根据CIE S 026/E标准，使用褪黑素作用谱计算mel-DER。
- **工具**：推荐使用Python的LuxPy库，支持这些参数的计算。
- **输出**：每个SPD对应的五个参数值。

##### 问题2：设计多通道LED系统
- **目标**：利用五个LED通道（蓝光、绿光、红光、暖白光、冷白光）设计满足特定场景的光谱。
- **场景**：
  - 日间照明：最大化Rf，确保CCT在6000±500 K，Rg在95-105之间，Rf>88。
  - 夜间助眠：最小化mel-DER，确保CCT在3000±500 K，Rf≥80。
- **方法**：
  - 将组合光谱表示为各通道SPD的加权线性叠加。
  - 针对每个场景，设置优化目标和约束条件，使用线性或非线性规划求解最优权重。
- **工具**：使用SciPy优化库，结合LuxPy计算参数。
- **输出**：每个场景的最优权重和关键参数（CCT, Duv, Rf, Rg, mel-DER）。

##### 问题3：模拟日光变化
- **目标**：利用五通道LED系统模拟一天中日光的谱变化。
- **方法**：
  - 对每个时间点（8:30至19:30）的日光SPD数据，优化LED通道权重以最小化与目标SPD的差异（例如最小二乘法）。
  - 选择早晨、正午和傍晚三个代表性时间点，绘制组合光谱与目标日光光谱的对比图。
- **工具**：使用SciPy优化，LuxPy进行光谱比较。
- **输出**：控制策略（权重随时间变化）和对比图。

##### 问题4：分析睡眠数据
- **目标**：评估优化光照是否显著改善睡眠质量。
- **方法**：
  - 计算睡眠质量指标：总睡眠时间（TST）、睡眠效率（SE）、入睡潜伏期（SOL）、深睡眠比例（N3%）、REM睡眠比例（REM%）和夜间醒来次数。
  - 使用统计方法（如ANOVA）比较三种光照环境（优化光、普通LED、黑暗）对指标的影响。
- **工具**：使用SciPy或R进行统计分析。
- **输出**：统计结果和结论。

---

---

#### 详细调研报告

##### 引言
本报告针对2025年华数杯全国大学生数学建模竞赛C题“可调控生物节律的LED光源研究”提供建模的详细思路和流程框架。该课题涉及LED光源的光谱特性优化，以满足照明需求并调节人体生理节律。问题分为四个部分：计算光谱参数、设计多通道LED系统、模拟日光变化和分析睡眠数据。以下是基于问题描述和数据分析的全面建模框架。

##### 数据概述
从附件“question.md”中提取问题详情，附件“data.xlsx”包含四个工作表：
- **Problem 1**：波长和光强数据，可能是单组LED的SPD。
- **Problem 2_LED_SPD**：五个LED通道（蓝光、绿光、红光、暖白光、冷白光）的SPD数据。
- **Problem 3_SUN_SPD**：从早晨8:30至傍晚19:30的日光SPD数据。
- **Problem 4**：11位被试在三种光照环境下的睡眠阶段数据。

这些数据为建模提供了基础，需结合标准方法和工具进行处理。

##### 问题1：计算光谱参数
###### 方法与工具
- **CCT和Duv**：根据IES TM-40和NIST文献（如Ohno, 2013），需将SPD转换为CIE XYZ值，再转换为CIE 1960 uv坐标，寻找最近的普朗克轨迹点计算CCT和Duv。相关资源包括：
  - [Waveform Lighting CCT Calculator](https://www.waveformlighting.com/tech/calculate-color-temperature-cct-from-cie-1931-xy-coordinates)
  - [NIST Practical Use and Calculation of CCT and Duv](https://www.nist.gov/publications/practical-use-and-calculation-cct-and-duv)
- **Rf和Rg**：采用IES TM-30-20方法，计算测试颜色集的颜色差异。标准文档可参考：
  - [IES TM-30-20](https://www.ies.org/store/technical-memoranda/)
- **mel-DER**：根据CIE S 026/E，使用褪黑素作用谱加权SPD计算。相关文献包括：
  - [CIE S 026/E](https://cie.co.at/publications/cie-s-026e-2018-melanopic-lux-and-spectral-melanopic-efficiency-function-for-light-and-health-applications)

- **工具选择**：LuxPy（Python库）支持上述所有计算，文档显示其包含`spd_to_xyz`、`xyz_to_cct`、`tm30`和`spd_to_mel`等函数，适合处理光谱数据。

###### 流程
1. 加载Problem 1工作表中的SPD数据。
2. 使用LuxPy计算CCT、Duv（通过XYZ转换）、Rf、Rg（TM-30-20方法）和mel-DER（加权积分）。
3. 输出计算结果。

##### 问题2：设计多通道LED系统
###### 建模思路
- 多通道光源的组合SPD为各通道SPD的加权线性叠加：  
  $$\text{Combined SPD} = w_1 \cdot \text{Blue} + w_2 \cdot \text{Green} + w_3 \cdot \text{Red} + w_4 \cdot \text{Warm White} + w_5 \cdot \text{Cold White}$$

- 需满足约束条件：
  - 日间照明：CCT ∈ [5500, 6500] K，Rg ∈ [95, 105]，Rf > 88，目标最大化Rf。
  - 夜间助眠：CCT ∈ [2500, 3500] K，Rf ≥ 80，目标最小化mel-DER。

###### 优化方法
- 将问题转化为优化问题：
  - 目标函数：日间为最大化Rf，夜间为最小化mel-DER。
  - 约束：权重和为1（$w_1 + w_2 + w_3 + w_4 + w_5 = 1$，权重非负），以及参数范围。
- 使用SciPy的优化模块（如`minimize`或`differential_evolution`）求解，结合LuxPy计算参数。

###### 流程
1. 加载Problem 2_LED_SPD工作表，获取五个通道的SPD。
2. 为每个场景设置优化问题：
   - 日间：目标函数为-Rf（最大化），约束为CCT范围和Rg范围。
   - 夜间：目标函数为mel-DER（最小化），约束为CCT范围和Rf下限。
3. 使用优化工具求解最优权重。
4. 计算并报告组合光谱的关键参数（CCT, Duv, Rf, Rg, mel-DER）。

##### 问题3：模拟日光变化
###### 建模思路
- 目标是使LED组合光谱接近给定时间点的日光SPD。优化目标为最小化二范数：  
  $$\text{Minimize} \sum (\text{Combined SPD} - \text{Sunlight SPD})^2$$

###### 流程
1. 加载Problem 3_SUN_SPD工作表，获取时间序列的日光SPD。
2. 对每个时间点，设置优化问题，目标为最小化组合SPD与目标SPD的差异。
3. 使用SciPy优化求解权重。
4. 选择早晨、正午、傍晚三个时间点，绘制组合光谱与目标日光光谱的对比图（可使用Matplotlib）。

##### 问题4：分析睡眠数据
###### 数据处理
- 数据为11位被试在三种环境（优化光、普通LED、黑暗）下的睡眠阶段记录，编码如下：
  | 编码 | 睡眠阶段       | 说明                     |
  |------|----------------|--------------------------|
  | 4    | 清醒 (Wake)    |                          |
  | 5    | REM睡眠        |                          |
  | 2    | 浅睡眠 (N1+N2) |                          |
  | 3    | 深睡眠 (N3)   | 慢波睡眠                 |

- 计算指标：
  | 指标         | 英文名称               | 计算方法                                   |
  |--------------|------------------------|--------------------------------------------|
  | 总睡眠时间   | Total Sleep Time (TST) | 非清醒阶段总时长                           |
  | 睡眠效率     | Sleep Efficiency (SE)  | $SE = \frac{TST}{\text{总卧床时间}} \times 100\%$ |
  | 深睡眠比例   | N3%                   | $N3\% = \frac{\text{N3期总时长}}{TST} \times 100\%$ |
  | REM睡眠比例  | REM%                  | $REM\% = \frac{\text{REM期总时长}}{TST} \times 100\%$ |
  | 夜间醒来次数 | Number of Awakenings  | 睡眠开始后清醒总次数                       |