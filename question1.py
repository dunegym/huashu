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

# 用McCamy近似公式法计算相关色温
def mccamy_calc_cct(xy):
    n = (xy[0] - 0.3320) / (xy[1] - 0.1858)
    return (-437 * n ** 3 + 3601 * n ** 2 - 6861 * n + 5514.31)

# 用三角垂足插值法(用日光轨迹近似黑体轨迹)计算相关色温
# def sun_triangle_calc_cct(xy):
#     sun_coordinates_dict = {}
    
#     # 生成日光轨迹坐标
#     for i in range(4000, 7001, 30):
#         x = -4.607e9 / (i ** 3) + 2967800 / (i ** 2) + 99.11 / i + 0.244063
#         y = -3 * x ** 2 + 2.87 * x - 0.275
#         sun_coordinates_dict[i] = (x, y)
#     for i in range(7030, 10001, 30):
#         x = -2.0064e9 / (i ** 3) + 1901800 / (i ** 2) + 247.48 / i + 0.23704
#         y = -3 * x ** 2 + 2.87 * x - 0.275
#         sun_coordinates_dict[i] = (x, y)
    
#     # 将字典转换为按温度排序的列表
#     temperatures = sorted(sun_coordinates_dict.keys())
    
#     min_distance_to_line = float('inf')
#     best_temp = 0
    
#     # 遍历相邻的两个点，找到与目标点距离最近的线段
#     for i in range(len(temperatures) - 1):
#         t1, t2 = temperatures[i], temperatures[i + 1]
#         p1 = sun_coordinates_dict[t1]  # (x1, y1)
#         p2 = sun_coordinates_dict[t2]  # (x2, y2)
        
#         # 计算线段p1p2上到目标点xy的垂足
#         x1, y1 = p1
#         x2, y2 = p2
#         x0, y0 = xy[0], xy[1]
        
#         # 线段方向向量
#         dx = x2 - x1
#         dy = y2 - y1
        
#         # 如果两点重合，跳过
#         if dx == 0 and dy == 0:
#             continue
            
#         # 计算参数t，表示垂足在线段上的位置
#         # 垂足公式：P = P1 + t * (P2 - P1)
#         t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        
#         # 限制t在[0,1]范围内，确保垂足在线段上
#         t = max(0, min(1, t))
        
#         # 计算垂足坐标
#         foot_x = x1 + t * dx
#         foot_y = y1 + t * dy
        
#         # 计算目标点到垂足的距离
#         distance = np.sqrt((x0 - foot_x) ** 2 + (y0 - foot_y) ** 2)
        
#         # 如果这是目前找到的最短距离，更新结果
#         if distance < min_distance_to_line:
#             min_distance_to_line = distance
#             # 使用线性插值计算对应的色温
#             best_temp = t1 + t * (t2 - t1)
    
#     return best_temp


# 用三角垂足插值法(通过普朗克公式计算黑体轨迹)计算相关色温
def blackbody_triangle_calc_cct(xy):
    """
    使用普朗克公式生成黑体轨迹，然后用三角垂足插值法计算相关色温
    :param xy: 目标色品坐标 (x, y)
    :return: 相关色温 CCT
    """
    # 物理常数
    h = 6.62607015e-34  # 普朗克常数 (J·s)
    c = 299792458       # 光速 (m/s)
    k_b = 1.380649e-23  # 玻尔兹曼常数 (J/K)
    
    # 波长范围 (nm)
    wavelengths = np.arange(380, 781, 5)  # 可见光范围，步长5nm
    
    # CIE 1931 2度标准观察者
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    
    blackbody_coordinates_dict = {}
    
    # 生成黑体轨迹坐标
    for temp in range(1000, 20001, 10):  # 温度范围1000K到20000K，步长10K
        # 计算该温度下的光谱功率分布
        spectral_data = {}
        
        for wavelength in wavelengths:
            # 将波长从nm转换为m
            lambda_m = wavelength * 1e-9
            
            # 普朗克公式计算光谱辐射亮度
            # B(λ,T) = (2hc²/λ⁵) * 1/(e^(hc/λkT) - 1)
            numerator = 2 * h * c**2 / (lambda_m**5)
            denominator = np.exp(h * c / (lambda_m * k_b * temp)) - 1
            spectral_radiance = numerator / denominator
            
            spectral_data[wavelength] = spectral_radiance
        
        # 创建光谱分布对象
        sd_blackbody = SpectralDistribution(spectral_data)
        
        try:
            # 计算三刺激值
            XYZ_blackbody = sd_to_XYZ(sd_blackbody, cmfs)
            
            # 计算色品坐标
            xy_blackbody = XYZ_to_xy(XYZ_blackbody)
            
            # 存储该温度下的色品坐标
            blackbody_coordinates_dict[temp] = (xy_blackbody[0], xy_blackbody[1])
            
        except Exception:
            # 如果计算失败，跳过这个温度点
            continue
    
    # 将字典转换为按温度排序的列表
    temperatures = sorted(blackbody_coordinates_dict.keys())
    
    min_distance_to_line = float('inf')
    best_temp = 0
    
    # 遍历相邻的两个点，找到与目标点距离最近的线段
    for i in range(len(temperatures) - 1):
        t1, t2 = temperatures[i], temperatures[i + 1]
        p1 = blackbody_coordinates_dict[t1]  # (x1, y1)
        p2 = blackbody_coordinates_dict[t2]  # (x2, y2)
        
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


def xy_to_uv(xy):
    """
    将CIE 1931色度坐标(x,y)转换为CIE 1976 UCS色度坐标(u',v')
    :param xy: 色度坐标 (x, y)
    :return: UCS色度坐标 (u', v')
    """
    x, y = xy[0], xy[1]
    denominator = -2 * x + 12 * y + 3
    
    # 避免除零错误
    if denominator == 0:
        return None
    
    u_prime = 4 * x / denominator
    v_prime = 9 * y / denominator
    
    return np.array([u_prime, v_prime])

def XYZ_to_uv(XYZ):
    """
    将三刺激值XYZ转换为CIE 1976 UCS色度坐标(u',v')
    :param XYZ: 三刺激值 (X, Y, Z)
    :return: UCS色度坐标 (u', v')
    """
    X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
    denominator = X + 15 * Y + 3 * Z
    
    # 避免除零错误
    if denominator == 0:
        return None
    
    u_prime = 4 * X / denominator
    v_prime = 9 * Y / denominator
    
    return np.array([u_prime, v_prime])


# 计算目标光源在uv色度图上与黑体轨迹的最短距离（色偏差）
def calc_color_deviation_uv(uv_target):
    """
    计算目标光源在uv色度图上与黑体轨迹的最短距离（色偏差）
    :param uv_target: 目标光源的uv坐标 (u', v')
    :return: 色偏差值（正值表示在黑体轨迹上方，负值表示在下方）、最近点的uv坐标、对应的色温
    """
    # 物理常数
    h = 6.62607015e-34  # 普朗克常数 (J·s)
    c = 299792458       # 光速 (m/s)
    k_b = 1.380649e-23  # 玻尔兹曼常数 (J/K)
    
    # 波长范围 (nm)
    wavelengths = np.arange(380, 781, 5)  # 可见光范围，步长5nm
    
    # CIE 1931 2度标准观察者
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    
    blackbody_uv_dict = {}
    
    # 生成黑体轨迹的uv坐标
    for temp in range(1000, 20001, 10):  # 温度范围1000K到20000K，步长10K
        # 计算该温度下的光谱功率分布
        spectral_data = {}
        
        for wavelength in wavelengths:
            # 将波长从nm转换为m
            lambda_m = wavelength * 1e-9
            
            # 普朗克公式计算光谱辐射亮度
            numerator = 2 * h * c**2 / (lambda_m**5)
            denominator = np.exp(h * c / (lambda_m * k_b * temp)) - 1
            spectral_radiance = numerator / denominator
            
            spectral_data[wavelength] = spectral_radiance
        
        # 创建光谱分布对象
        sd_blackbody = SpectralDistribution(spectral_data)
        
        try:
            # 计算三刺激值
            XYZ_blackbody = sd_to_XYZ(sd_blackbody, cmfs)
            
            # 计算uv坐标
            uv_blackbody = XYZ_to_uv(XYZ_blackbody)
            
            if uv_blackbody is not None:
                blackbody_uv_dict[temp] = (uv_blackbody[0], uv_blackbody[1])
            
        except Exception:
            continue
    
    # 将字典转换为按温度排序的列表
    temperatures = sorted(blackbody_uv_dict.keys())
    
    min_distance = float('inf')
    best_temp = 0
    closest_point = None
    
    u_target, v_target = uv_target[0], uv_target[1]
    
    # 遍历相邻的两个点，找到与目标点距离最近的线段
    for i in range(len(temperatures) - 1):
        t1, t2 = temperatures[i], temperatures[i + 1]
        p1 = blackbody_uv_dict[t1]  # (u1, v1)
        p2 = blackbody_uv_dict[t2]  # (u2, v2)
        
        u1, v1 = p1
        u2, v2 = p2
        
        # 线段方向向量
        du = u2 - u1
        dv = v2 - v1
        
        # 如果两点重合，跳过
        if du == 0 and dv == 0:
            continue
            
        # 计算参数t，表示垂足在线段上的位置
        t = ((u_target - u1) * du + (v_target - v1) * dv) / (du * du + dv * dv)
        
        # 限制t在[0,1]范围内，确保垂足在线段上
        t = max(0, min(1, t))
        
        # 计算垂足坐标
        foot_u = u1 + t * du
        foot_v = v1 + t * dv
        
        # 计算目标点到垂足的距离
        distance = np.sqrt((u_target - foot_u) ** 2 + (v_target - foot_v) ** 2)
        
        # 如果这是目前找到的最短距离，更新结果
        if distance < min_distance:
            min_distance = distance
            closest_point = (foot_u, foot_v)
            # 使用线性插值计算对应的色温
            best_temp = t1 + t * (t2 - t1)
    
    # 确定距离的正负号
    # 需要判断目标点相对于黑体轨迹的位置
    # 在uv色度图中，一般可以通过比较v坐标来判断上下关系
    
    # 计算从最近点到目标点的向量
    if closest_point is not None:
        vec_u = u_target - closest_point[0]
        vec_v = v_target - closest_point[1]
        
        # 计算黑体轨迹在最近点处的切线方向
        # 找到最近点对应的温度区间
        for i in range(len(temperatures) - 1):
            t1, t2 = temperatures[i], temperatures[i + 1]
            p1 = blackbody_uv_dict[t1]
            p2 = blackbody_uv_dict[t2]
            
            # 检查最近点是否在这个线段上
            u1, v1 = p1
            u2, v2 = p2
            du = u2 - u1
            dv = v2 - v1
            
            if du == 0 and dv == 0:
                continue
                
            t = ((closest_point[0] - u1) * du + (closest_point[1] - v1) * dv) / (du * du + dv * dv)
            
            if 0 <= t <= 1:
                # 切线方向向量
                tangent_u = du
                tangent_v = dv
                
                # 计算法向量（垂直于切线，指向轨迹"上方"）
                normal_u = -tangent_v
                normal_v = tangent_u
                
                # 计算目标点相对于最近点的向量与法向量的点积
                dot_product = vec_u * normal_u + vec_v * normal_v
                
                # 如果点积为正，表示在轨迹上方；为负表示在轨迹下方
                signed_distance = min_distance if dot_product > 0 else -min_distance
                
                return signed_distance, closest_point, best_temp
    
    return min_distance, closest_point, best_temp

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
    xy_coordinates = XYZ_to_xy(tristimulus_values)
    print(f"计算得到的xy色品坐标 (xy) 为: {xy_coordinates}")
    
    uv_coordinates = xy_to_uv(xy_coordinates)
    print(f"计算得到的uv色度坐标 (u'v') 为: {uv_coordinates}")
    
    # 2.1 计算色偏差
    color_deviation, closest_point, closest_temp = calc_color_deviation_uv(uv_coordinates)
    print(f"目标光源与黑体轨迹的色偏差: {color_deviation:.6f}")
    print(f"黑体轨迹上最近点的uv坐标: {closest_point}")
    print(f"在1960uv色度图上对应的色温: {closest_temp}")
    
    # 3. 计算相对色温
    # CCT = colour.xy_to_CCT(xy_coordinates, method='McCamy 1992')
    # print(f"库函数计算的CCT: {CCT}") # 库函数的计算结果与论文中提供的算式得到的结果不同，故舍弃
    print(f"用McCamy近似公式法计算的CCT: {mccamy_calc_cct(xy_coordinates)}")
    # print(f"用日光轨迹三角垂足插值法计算的CCT: {sun_triangle_calc_cct(xy_coordinates)}")
    print(f"在1931xy色度图上用黑体轨迹三角垂足插值法计算的CCT: {blackbody_triangle_calc_cct(xy_coordinates)}")
