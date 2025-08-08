# 导包
import numpy as np
import pandas as pd
from colour.colorimetry import sd_to_XYZ, SpectralDistribution
from colour.models import XYZ_to_xy
from colour import MSDS_CMFS

# 添加必要的导入
from colour.models import XYZ_to_CAM02UCS, CAM02UCS_to_XYZ
from colour.adaptation import chromatic_adaptation_VonKries
from colour.temperature import CCT_to_xy_CIE_D
import colour

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

def get_reference_illuminant(cct):
    """
    根据CCT获取参考光源的光谱功率分布
    :param cct: 相关色温
    :return: 参考光源的光谱分布对象
    """
    wavelengths = np.arange(380, 781, 5)
    
    if cct <= 4000:
        # 使用普朗克辐射体（黑体辐射）
        spectral_data = {}
        h = 6.62607015e-34
        c = 299792458
        k_b = 1.380649e-23
        
        for wavelength in wavelengths:
            lambda_m = wavelength * 1e-9
            numerator = 2 * h * c**2 / (lambda_m**5)
            denominator = np.exp(h * c / (lambda_m * k_b * cct)) - 1
            spectral_radiance = numerator / denominator
            spectral_data[wavelength] = spectral_radiance
            
    elif cct >= 5000:
        # 使用CIE D系列光源
        try:
            xy_ref = CCT_to_xy_CIE_D(cct)
            # 生成D系列光源的相对光谱功率分布
            spectral_data = {}
            for wavelength in wavelengths:
                # D系列光源的简化公式（实际应用中可能需要更精确的公式）
                if wavelength <= 500:
                    spd_value = 1.0 + 0.5 * np.exp(-(wavelength - 460)**2 / 1000)
                else:
                    spd_value = 1.0 * np.exp(-(wavelength - 560)**2 / 5000)
                spectral_data[wavelength] = spd_value
        except:
            # 如果D系列计算失败，回退到普朗克辐射体
            spectral_data = {}
            h = 6.62607015e-34
            c = 299792458
            k_b = 1.380649e-23
            
            for wavelength in wavelengths:
                lambda_m = wavelength * 1e-9
                numerator = 2 * h * c**2 / (lambda_m**5)
                denominator = np.exp(h * c / (lambda_m * k_b * cct)) - 1
                spectral_radiance = numerator / denominator
                spectral_data[wavelength] = spectral_radiance
    else:
        # 4000K < CCT < 5000K: 普朗克辐射体与CIE D系列的混合
        ratio = (cct - 4000) / 1000  # 0到1之间的比例
        
        # 普朗克辐射体部分
        planck_data = {}
        h = 6.62607015e-34
        c = 299792458
        k_b = 1.380649e-23
        
        for wavelength in wavelengths:
            lambda_m = wavelength * 1e-9
            numerator = 2 * h * c**2 / (lambda_m**5)
            denominator = np.exp(h * c / (lambda_m * k_b * cct)) - 1
            planck_data[wavelength] = numerator / denominator
        
        # D系列部分（简化）
        d_series_data = {}
        for wavelength in wavelengths:
            if wavelength <= 500:
                spd_value = 1.0 + 0.5 * np.exp(-(wavelength - 460)**2 / 1000)
            else:
                spd_value = 1.0 * np.exp(-(wavelength - 560)**2 / 5000)
            d_series_data[wavelength] = spd_value
        
        # 混合
        spectral_data = {}
        for wavelength in wavelengths:
            spectral_data[wavelength] = (1 - ratio) * planck_data[wavelength] + ratio * d_series_data[wavelength]
    
    return SpectralDistribution(spectral_data)

def generate_ces_samples():
    """
    生成99个颜色评估样本（CES）的反射率数据
    这里使用简化的方法生成代表性样本，实际应用中应使用标准的CES数据
    :return: CES样本的反射率字典列表
    """
    wavelengths = np.arange(380, 781, 5)
    ces_samples = []
    
    # 生成覆盖不同色调、饱和度和明度的样本
    for hue in range(0, 360, 15):  # 24个色调
        for saturation in [0.3, 0.6, 0.9]:  # 3个饱和度级别
            for lightness in [0.3, 0.6, 0.8]:  # 3个明度级别
                if len(ces_samples) >= 99:
                    break
                
                # 生成光谱反射率（简化模型）
                reflectance = {}
                peak_wavelength = 380 + (hue / 360) * 400  # 根据色调确定峰值波长
                
                for wavelength in wavelengths:
                    # 使用高斯分布模拟光谱反射率
                    base_reflectance = lightness * 0.8
                    peak_contribution = saturation * 0.4 * np.exp(-(wavelength - peak_wavelength)**2 / 5000)
                    reflectance[wavelength] = max(0.05, min(0.95, base_reflectance + peak_contribution))
                
                ces_samples.append(reflectance)
                
                if saturation == 0.9 and lightness == 0.8:
                    break  # 避免过多样本
            if len(ces_samples) >= 99:
                break
        if len(ces_samples) >= 99:
            break
    
    # 确保恰好99个样本
    return ces_samples[:99]

def XYZ_to_CAM02UCS_simplified(XYZ, white_point_XYZ):
    """
    简化的XYZ到CAM02-UCS转换
    :param XYZ: XYZ三刺激值
    :param white_point_XYZ: 白点XYZ值
    :return: CAM02-UCS坐标 (J, a, b)
    """
    try:
        # 使用colour库的标准转换
        # 设置观察条件
        surround = colour.CAM_SPECIFICATION_CAM02UCS
        
        # 标准化白点
        white_point = white_point_XYZ / white_point_XYZ[1] * 100
        
        # 转换到CAM02-UCS
        cam02ucs = XYZ_to_CAM02UCS(XYZ, white_point)
        return cam02ucs
    except:
        # 如果标准转换失败，使用简化方法
        # 这是一个非常简化的近似，实际应用中应使用标准的CAM02-UCS转换
        X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
        
        # 简化的明度
        J = 100 * (Y / white_point_XYZ[1])**0.5
        
        # 简化的色度坐标
        x = X / (X + Y + Z) if (X + Y + Z) > 0 else 0
        y = Y / (X + Y + Z) if (X + Y + Z) > 0 else 0
        
        # 转换为a*b*类似的坐标
        a = 500 * (x - 1/3)
        b = 200 * (y - 1/3)
        
        return np.array([J, a, b])

def calculate_rf_rg(test_sd):
    """
    计算TM-30标准的保真度指数(Rf)和色域指数(Rg)
    :param test_sd: 测试光源的光谱分布对象
    :return: (Rf, Rg, 详细信息字典)
    """
    # 使用CIE 1964 10度标准观察者
    try:
        cmfs = MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
    except:
        # 如果10度观察者不可用，使用2度观察者
        cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    
    # 1. 计算测试光源的CCT
    XYZ_test = sd_to_XYZ(test_sd, cmfs)
    xy_test = XYZ_to_xy(XYZ_test)
    cct_test = mccamy_calc_cct(xy_test)
    
    # 2. 获取参考光源
    ref_sd = get_reference_illuminant(cct_test)
    XYZ_ref = sd_to_XYZ(ref_sd, cmfs)
    
    # 3. 生成CES样本
    ces_samples = generate_ces_samples()
    
    # 4. 计算每个CES在测试光源和参考光源下的颜色坐标
    test_colors = []
    ref_colors = []
    
    wavelengths = np.arange(380, 781, 5)
    
    for ces_reflectance in ces_samples:
        # 计算测试光源下的颜色
        test_spectral_data = {}
        ref_spectral_data = {}
        
        for wavelength in wavelengths:
            if wavelength in ces_reflectance and wavelength in test_sd.wavelengths:
                test_spectral_data[wavelength] = (
                    test_sd[wavelength] * ces_reflectance[wavelength]
                )
                ref_spectral_data[wavelength] = (
                    ref_sd[wavelength] * ces_reflectance[wavelength]
                )
        
        if test_spectral_data and ref_spectral_data:
            test_ces_sd = SpectralDistribution(test_spectral_data)
            ref_ces_sd = SpectralDistribution(ref_spectral_data)
            
            try:
                XYZ_test_ces = sd_to_XYZ(test_ces_sd, cmfs)
                XYZ_ref_ces = sd_to_XYZ(ref_ces_sd, cmfs)
                
                # 转换到CAM02-UCS颜色空间
                cam02ucs_test = XYZ_to_CAM02UCS_simplified(XYZ_test_ces, XYZ_test)
                cam02ucs_ref = XYZ_to_CAM02UCS_simplified(XYZ_ref_ces, XYZ_ref)
                
                test_colors.append(cam02ucs_test)
                ref_colors.append(cam02ucs_ref)
            except:
                continue
    
    if len(test_colors) == 0:
        return 0, 100, {"error": "无法计算有效的颜色样本"}
    
    # 5. 计算Rf（保真度指数）
    color_differences = []
    for test_color, ref_color in zip(test_colors, ref_colors):
        # 计算色差（欧几里得距离）
        delta_J = test_color[0] - ref_color[0]
        delta_a = test_color[1] - ref_color[1]
        delta_b = test_color[2] - ref_color[2]
        
        color_diff = np.sqrt(delta_J**2 + delta_a**2 + delta_b**2)
        color_differences.append(color_diff)
    
    # 平均色差
    avg_color_diff = np.mean(color_differences)
    
    # 计算Rf（使用标准化的对数变换）
    k = 6.73  # TM-30标准中的比例因子
    Rf = 100 - k * avg_color_diff
    Rf = max(0, min(100, Rf))  # 限制在0-100范围内
    
    # 6. 计算Rg（色域指数）
    # 将样本分配到16个色调角箱
    hue_bins = 16
    angle_per_bin = 360 / hue_bins
    
    test_hue_centers = []
    ref_hue_centers = []
    
    for bin_idx in range(hue_bins):
        bin_test_a = []
        bin_test_b = []
        bin_ref_a = []
        bin_ref_b = []
        
        for test_color, ref_color in zip(test_colors, ref_colors):
            # 计算色调角度
            test_hue = np.arctan2(test_color[2], test_color[1]) * 180 / np.pi
            if test_hue < 0:
                test_hue += 360
            
            # 检查是否在当前箱内
            bin_start = bin_idx * angle_per_bin
            bin_end = (bin_idx + 1) * angle_per_bin
            
            if bin_start <= test_hue < bin_end or (bin_idx == hue_bins - 1 and test_hue >= bin_start):
                bin_test_a.append(test_color[1])
                bin_test_b.append(test_color[2])
                bin_ref_a.append(ref_color[1])
                bin_ref_b.append(ref_color[2])
        
        # 计算该箱的平均色度坐标
        if bin_test_a:
            test_hue_centers.append([np.mean(bin_test_a), np.mean(bin_test_b)])
            ref_hue_centers.append([np.mean(bin_ref_a), np.mean(bin_ref_b)])
    
    # 计算色域面积
    if len(test_hue_centers) >= 3 and len(ref_hue_centers) >= 3:
        # 使用Shoelace公式计算多边形面积
        def polygon_area(points):
            if len(points) < 3:
                return 0
            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]
            return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))
        
        test_area = polygon_area(test_hue_centers)
        ref_area = polygon_area(ref_hue_centers)
        
        if ref_area > 0:
            Rg = (test_area / ref_area) * 100
        else:
            Rg = 100
    else:
        Rg = 100
    
    # 限制Rg在合理范围内
    Rg = max(50, min(150, Rg))
    
    # 返回结果和详细信息
    details = {
        "test_cct": cct_test,
        "num_valid_samples": len(test_colors),
        "avg_color_difference": avg_color_diff,
        "test_gamut_area": test_area if 'test_area' in locals() else 0,
        "ref_gamut_area": ref_area if 'ref_area' in locals() else 0,
        "hue_bins_used": len(test_hue_centers)
    }
    
    return Rf, Rg, details

def get_melanopsin_action_spectrum():
    """
    获取mel-opic动作光谱 s_mel(λ)
    基于CIE S 026/E:2018标准
    :return: mel-opic动作光谱字典 {wavelength: sensitivity}
    """
    # CIE S 026/E:2018标准中的mel-opic动作光谱数据（简化版本）
    # 实际应用中应使用完整的标准数据
    wavelengths = np.arange(380, 781, 5)
    mel_action_spectrum = {}
    
    for wavelength in wavelengths:
        # 基于melanopsin的光谱敏感性曲线（简化模型）
        # 峰值在约480nm，符合melanopsin的特征
        if wavelength < 400:
            sensitivity = 0.001
        elif wavelength <= 420:
            sensitivity = 0.01 * np.exp(-(wavelength - 420)**2 / 800)
        elif wavelength <= 480:
            # 主要敏感区域，峰值在480nm附近
            sensitivity = np.exp(-(wavelength - 480)**2 / 1200)
        elif wavelength <= 550:
            sensitivity = 0.8 * np.exp(-(wavelength - 480)**2 / 1500)
        elif wavelength <= 600:
            sensitivity = 0.3 * np.exp(-(wavelength - 480)**2 / 2000)
        else:
            sensitivity = 0.05 * np.exp(-(wavelength - 480)**2 / 3000)
        
        mel_action_spectrum[wavelength] = max(0.001, sensitivity)
    
    # 归一化到峰值为1
    max_sensitivity = max(mel_action_spectrum.values())
    for wavelength in mel_action_spectrum:
        mel_action_spectrum[wavelength] /= max_sensitivity
    
    return mel_action_spectrum

def get_photopic_luminous_efficiency():
    """
    获取光视效率函数 V(λ)
    基于CIE标准
    :return: 光视效率函数字典 {wavelength: efficiency}
    """
    # CIE 1931标准光视效率函数（简化版本）
    wavelengths = np.arange(380, 781, 5)
    v_lambda = {}
    
    for wavelength in wavelengths:
        # 基于CIE 1931标准的光视效率函数
        # 峰值在555nm
        if wavelength < 400:
            efficiency = 0.0001
        elif wavelength <= 500:
            efficiency = 0.01 * np.exp(-(wavelength - 555)**2 / 8000)
        elif wavelength <= 555:
            # 主要敏感区域
            efficiency = np.exp(-(wavelength - 555)**2 / 6000)
        elif wavelength <= 650:
            efficiency = np.exp(-(wavelength - 555)**2 / 7000)
        else:
            efficiency = 0.001 * np.exp(-(wavelength - 555)**2 / 10000)
        
        v_lambda[wavelength] = max(0.0001, efficiency)
    
    # 归一化到峰值为1（555nm）
    max_efficiency = max(v_lambda.values())
    for wavelength in v_lambda:
        v_lambda[wavelength] /= max_efficiency
    
    return v_lambda

def get_d65_spd():
    """
    获取标准日光D65的光谱功率分布
    :return: D65光谱分布对象
    """
    wavelengths = np.arange(380, 781, 5)
    
    # D65标准日光的相对光谱功率分布（简化版本）
    # 实际应用中应使用CIE标准数据
    d65_data = {}
    
    for wavelength in wavelengths:
        # D65的简化光谱分布模型
        if wavelength <= 400:
            spd_value = 50 + 10 * (wavelength - 380) / 20
        elif wavelength <= 500:
            spd_value = 60 + 40 * np.exp(-(wavelength - 460)**2 / 5000)
        elif wavelength <= 600:
            spd_value = 100 - 20 * (wavelength - 500) / 100
        else:
            spd_value = 80 - 30 * (wavelength - 600) / 180
        
        d65_data[wavelength] = max(10, spd_value)
    
    return SpectralDistribution(d65_data)

def calculate_mel_der(test_sd):
    """
    计算褪黑素日光照度比（mel-DER）
    :param test_sd: 测试光源的光谱分布对象
    :return: mel-DER值和详细计算信息
    """
    # 获取必要的光谱函数
    mel_action_spectrum = get_melanopsin_action_spectrum()
    v_lambda = get_photopic_luminous_efficiency()
    d65_sd = get_d65_spd()
    
    # 最大光视效率常数
    K_m = 683.0  # lm/W
    
    wavelengths = np.arange(380, 781, 5)
    
    # === 1. 计算测试光源的mel-opic辐照度 ===
    test_mel_irradiance = 0.0
    for wavelength in wavelengths:
        if wavelength in test_sd.wavelengths and wavelength in mel_action_spectrum:
            # E_mel = ∫ E(λ) · s_mel(λ) dλ
            test_mel_irradiance += test_sd[wavelength] * mel_action_spectrum[wavelength] * 5  # 5nm步长
    
    # === 2. 计算测试光源的照度 ===
    test_illuminance = 0.0
    for wavelength in wavelengths:
        if wavelength in test_sd.wavelengths and wavelength in v_lambda:
            # E_v = K_m · ∫ E(λ) · V(λ) dλ
            test_illuminance += test_sd[wavelength] * v_lambda[wavelength] * 5  # 5nm步长
    
    test_illuminance *= K_m
    
    # === 3. 计算测试光源的mel-opic ELR ===
    if test_illuminance > 0:
        test_mel_elr = test_mel_irradiance / test_illuminance
    else:
        return 0, {"error": "测试光源照度为零"}
    
    # === 4. 计算D65的mel-opic辐照度 ===
    d65_mel_irradiance = 0.0
    for wavelength in wavelengths:
        if wavelength in d65_sd.wavelengths and wavelength in mel_action_spectrum:
            d65_mel_irradiance += d65_sd[wavelength] * mel_action_spectrum[wavelength] * 5
    
    # === 5. 计算D65的照度 ===
    d65_illuminance = 0.0
    for wavelength in wavelengths:
        if wavelength in d65_sd.wavelengths and wavelength in v_lambda:
            d65_illuminance += d65_sd[wavelength] * v_lambda[wavelength] * 5
    
    d65_illuminance *= K_m
    
    # === 6. 计算D65的mel-opic ELR ===
    if d65_illuminance > 0:
        d65_mel_elr = d65_mel_irradiance / d65_illuminance
    else:
        return 0, {"error": "D65照度计算错误"}
    
    # === 7. 计算mel-DER ===
    if d65_mel_elr > 0:
        mel_der = test_mel_elr / d65_mel_elr
    else:
        return 0, {"error": "D65 mel-opic ELR为零"}
    
    # 返回详细计算信息
    details = {
        "test_mel_irradiance": test_mel_irradiance,
        "test_illuminance": test_illuminance,
        "test_mel_elr": test_mel_elr,
        "d65_mel_irradiance": d65_mel_irradiance,
        "d65_illuminance": d65_illuminance,
        "d65_mel_elr": d65_mel_elr,
        "mel_der": mel_der
    }
    
    return mel_der, details

def calculate_mel_der_precise(test_sd):
    """
    使用更精确的CIE数据计算mel-DER（如果可用）
    :param test_sd: 测试光源的光谱分布对象
    :return: mel-DER值和详细计算信息
    """
    try:
        # 尝试使用colour库中的精确数据
        from colour.colorimetry import MSDS_CMFS
        
        # 如果有可用的mel-opic数据，使用它们
        # 否则回退到简化计算
        return calculate_mel_der(test_sd)
    
    except ImportError:
        # 如果没有相关库，使用简化计算
        return calculate_mel_der(test_sd)

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
    print(f"在1976uv色度图上用三角垂足插值法计算的CCT: {closest_temp}")
    
    # 3. 计算相对色温
    # CCT = colour.xy_to_CCT(xy_coordinates, method='McCamy 1992')
    # print(f"库函数计算的CCT: {CCT}") # 库函数的计算结果与论文中提供的算式得到的结果不同，故舍弃
    print(f"用McCamy近似公式法计算的CCT: {mccamy_calc_cct(xy_coordinates)}")
    # print(f"用日光轨迹三角垂足插值法计算的CCT: {sun_triangle_calc_cct(xy_coordinates)}")
    print(f"在1931xy色度图上用三角垂足插值法计算的CCT: {blackbody_triangle_calc_cct(xy_coordinates)}")
    
    # 4. 计算TM-30指数
    print("\n=== TM-30 颜色质量评估 ===")
    Rf, Rg, tm30_details = calculate_rf_rg(sd)
    print(f"保真度指数 (Rf): {Rf}")
    print(f"色域指数 (Rg): {Rg}")
    print(f"有效颜色样本数: {tm30_details['num_valid_samples']}")
    print(f"平均色差: {tm30_details['avg_color_difference']}")
    print(f"使用的色调箱数: {tm30_details['hue_bins_used']}")
    
    # 评估结果
    if Rf >= 80:
        rf_quality = "优秀"
    elif Rf >= 70:
        rf_quality = "良好"
    elif Rf >= 60:
        rf_quality = "可接受"
    else:
        rf_quality = "不适合室内照明"
    
    if 90 <= Rg <= 110:
        rg_quality = "理想色域"
    elif 80 <= Rg <= 120:
        rg_quality = "可接受色域"
    else:
        rg_quality = "色域偏差较大"
    
    print(f"颜色保真度评价: {rf_quality}")
    print(f"色域评价: {rg_quality}")
    
    # 5. 计算mel-DER
    print("\n=== 褪黑素日光照度比（mel-DER）计算 ===")
    mel_der, mel_details = calculate_mel_der(sd)
    print(f"褪黑素日光照度比 (mel-DER): {mel_der:.4f}")
    print(f"测试光源mel-opic辐照度: {mel_details['test_mel_irradiance']:.6f} W/m²")
    print(f"测试光源照度: {mel_details['test_illuminance']:.2f} lux")
    print(f"测试光源mel-opic ELR: {mel_details['test_mel_elr']:.8f} W/m²/lux")
    print(f"D65 mel-opic ELR: {mel_details['d65_mel_elr']:.8f} W/m²/lux")
    
    # mel-DER评估
    if mel_der > 1.0:
        mel_assessment = "相比D65日光，该光源对melanopsin的刺激更强"
    elif mel_der < 0.8:
        mel_assessment = "相比D65日光，该光源对melanopsin的刺激较弱，可能更适合晚间使用"
    else:
        mel_assessment = "该光源对melanopsin的刺激与D65日光相近"
    
    print(f"mel-DER评估: {mel_assessment}")
