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

# 用McCamy近似公式法计算相对色温（CCT）
def mccamy_calc_cct(xy):
    n = (xy[0] - 0.3320) / (xy[1] - 0.1858)
    return (-437 * n ** 3 + 3601 * n ** 2 - 6861 * n + 5514.31)


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
