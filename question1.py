# 导包
import numpy as np
import pandas as pd
from colour.colorimetry import sd_to_XYZ, SpectralDistribution
from colour.models import XYZ_to_xy
from colour import MSDS_CMFS

# 计算三刺激值
def calc_tsv(sd):
    """
    根据光谱分布计算三刺激值 XYZ.
    :param sd: 光谱分布对象
    :return: 三刺激值 (X, Y, Z)
    """
    # 使用CIE 1931 2度标准观察者
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    # 计算三刺激值
    XYZ = sd_to_XYZ(sd, cmfs)
    return XYZ
    

# 计算色品坐标
def calc_coordinates(XYZ):
    """
    根据三刺激值 XYZ 计算色品坐标 xy.
    :param XYZ: 三刺激值
    :return: 色品坐标 (x, y)
    """
    xy = XYZ_to_xy(XYZ)
    return xy

# 用McCamy近似公式法计算相关色温
def mccamy_calc_cct(xy):
    n = (xy[0] - 0.3320) / (xy[1] - 0.1858)
    return (-437 * n ** 3 + 3601 * n ** 2 - 6861 * n + 5514.31)

# 用三角垂足插值法计算相关色温
def triangle_calc_cct(xy):
    sun_coordinates_dict = {}
    
    # 生成黑体轨迹坐标
    for i in range(4000, 7001, 30):
        x = -4.607e9 / (i ** 3) + 2967800 / (i ** 2) + 99.11 / i + 0.244063
        y = -3 * x ** 2 + 2.87 * x - 0.275
        sun_coordinates_dict[i] = (x, y)
    for i in range(7030, 10001, 30):
        x = -2.0064e9 / (i ** 3) + 1901800 / (i ** 2) + 247.48 / i + 0.23704
        y = -3 * x ** 2 + 2.87 * x - 0.275
        sun_coordinates_dict[i] = (x, y)
    
    # 将字典转换为按温度排序的列表
    temperatures = sorted(sun_coordinates_dict.keys())
    
    min_distance_to_line = float('inf')
    best_temp = 0
    
    # 遍历相邻的两个点，找到与目标点距离最近的线段
    for i in range(len(temperatures) - 1):
        t1, t2 = temperatures[i], temperatures[i + 1]
        p1 = sun_coordinates_dict[t1]  # (x1, y1)
        p2 = sun_coordinates_dict[t2]  # (x2, y2)
        
        # 计算线段p1p2上到目标点xy的垂足
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = xy[0], xy[1]
        
        # 线段方向向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 如果两点重合，跳过
        if dx == 0 and dy == 0:
            continue
            
        # 计算参数t，表示垂足在线段上的位置
        # 垂足公式：P = P1 + t * (P2 - P1)
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        
        # 限制t在[0,1]范围内，确保垂足在线段上
        t = max(0, min(1, t))
        
        # 计算垂足坐标
        foot_x = x1 + t * dx
        foot_y = y1 + t * dy
        
        # 计算目标点到垂足的距离
        distance = np.sqrt((x0 - foot_x) ** 2 + (y0 - foot_y) ** 2)
        
        # 如果这是目前找到的最短距离，更新结果
        if distance < min_distance_to_line:
            min_distance_to_line = distance
            # 使用线性插值计算对应的色温
            best_temp = t1 + t * (t2 - t1)
    
    return best_temp


# 主程序
if __name__ =='__main__':
    # 读取波长与光强进np数组，波长为整型，光强为双精度浮点型
    df_1 = pd.read_excel('data.xlsx', sheet_name='Problem 1')
    wavelength_arr = np.array([int(str(x)[:3]) for x in df_1['波长'].to_numpy()])
    energy_arr = df_1['光强'].to_numpy().astype(np.float64)

    # 将波长和能量数据组合成colour-science库所需的格式
    spectral_data = dict(zip(wavelength_arr, energy_arr))
    sd = SpectralDistribution(spectral_data)

    # 1. 计算三刺激值
    tristimulus_values = calc_tsv(sd)
    print(f"计算得到的三刺激值 (XYZ) 为: {tristimulus_values}")

    # 2. 计算色品坐标
    chromaticity_coordinates = calc_coordinates(tristimulus_values)
    print(f"计算得到的色品坐标 (xy) 为: {chromaticity_coordinates}")
    
    # 3. 计算相对色温
    # CCT = colour.xy_to_CCT(chromaticity_coordinates, method='McCamy 1992')
    # print(f"库函数计算的CCT: {CCT}") # 库函数的计算结果与论文中提供的算式得到的结果不同，故舍弃
    print(f"用McCamy近似公式法计算的CCT: {mccamy_calc_cct(chromaticity_coordinates)}")
    print(f"用三角垂足插值法计算的CCT: {triangle_calc_cct(chromaticity_coordinates)}")
