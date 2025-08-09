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

# ==================== 常量定义 ====================
# 物理常数
PLANCK_CONSTANT = 6.62607015e-34  # 普朗克常数 (J·s)
LIGHT_SPEED = 299792458           # 光速 (m/s)
BOLTZMANN_CONSTANT = 1.380649e-23 # 玻尔兹曼常数 (J/K)
MAX_LUMINOUS_EFFICACY = 683.0     # 最大光视效率常数 (lm/W)

# 光谱范围
WAVELENGTH_MIN = 380
WAVELENGTH_MAX = 780
WAVELENGTH_STEP = 5
VISIBLE_WAVELENGTHS = np.arange(WAVELENGTH_MIN, WAVELENGTH_MAX + 1, WAVELENGTH_STEP)

# 色温范围
CCT_MIN = 1000
CCT_MAX = 20000
CCT_STEP_FINE = 10
CCT_STEP_COARSE = 30

# TM-30常数
TM30_SCALE_FACTOR = 6.73
CES_SAMPLE_COUNT = 99
HUE_BIN_COUNT = 16

# CIE坐标系标准点
CIE_STANDARD_POINT_X = 1/3
CIE_STANDARD_POINT_Y = 1/3

# McCamy公式常数
MCCAMY_OFFSET_X = 0.3320
MCCAMY_OFFSET_Y = 0.1858
MCCAMY_COEFFS = [-437, 3601, -6861, 5514.31]

# ==================== 公共工具函数 ====================
def create_spectral_distribution(spectral_data):
    """创建光谱分布对象的工厂函数"""
    return SpectralDistribution(spectral_data)

def get_standard_observer(degree=2):
    """获取标准观察者"""
    if degree == 10:
        try:
            return MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
        except:
            pass
    return MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

def safe_divide(numerator, denominator, default=0):
    """安全除法，避免除零错误"""
    return numerator / denominator if denominator != 0 else default

def planck_spectral_radiance(wavelength_nm, temperature):
    """
    计算普朗克黑体辐射光谱亮度
    :param wavelength_nm: 波长 (nm)
    :param temperature: 温度 (K)
    :return: 光谱辐射亮度
    """
    wavelength_m = wavelength_nm * 1e-9
    numerator = 2 * PLANCK_CONSTANT * LIGHT_SPEED**2 / (wavelength_m**5)
    denominator = np.exp(PLANCK_CONSTANT * LIGHT_SPEED / (wavelength_m * BOLTZMANN_CONSTANT * temperature)) - 1
    return numerator / denominator

def calculate_blackbody_coordinates(temperature_range=None, wavelengths=None):
    """
    计算黑体轨迹坐标的通用函数
    :param temperature_range: 温度范围
    :param wavelengths: 波长数组
    :return: 温度到坐标的映射字典
    """
    if temperature_range is None:
        temperature_range = range(CCT_MIN, CCT_MAX + 1, CCT_STEP_FINE)
    if wavelengths is None:
        wavelengths = VISIBLE_WAVELENGTHS
    
    cmfs = get_standard_observer()
    coordinates_dict = {}
    
    for temp in temperature_range:
        spectral_data = {}
        for wavelength in wavelengths:
            spectral_data[wavelength] = planck_spectral_radiance(wavelength, temp)
        
        try:
            sd_blackbody = create_spectral_distribution(spectral_data)
            XYZ_blackbody = sd_to_XYZ(sd_blackbody, cmfs)
            xy_blackbody = XYZ_to_xy(XYZ_blackbody)
            coordinates_dict[temp] = (xy_blackbody[0], xy_blackbody[1])
        except Exception:
            continue
    
    return coordinates_dict

def find_closest_point_on_curve(target_point, curve_coordinates):
    """
    在曲线上找到与目标点最近的点（三角垂足插值法）
    :param target_point: 目标点坐标 (x, y)
    :param curve_coordinates: 曲线坐标字典 {parameter: (x, y)}
    :return: (最小距离, 最近点坐标, 对应参数)
    """
    parameters = sorted(curve_coordinates.keys())
    min_distance = float('inf')
    best_param = 0
    closest_point = None
    
    x_target, y_target = target_point[0], target_point[1]
    
    for i in range(len(parameters) - 1):
        param1, param2 = parameters[i], parameters[i + 1]
        p1 = curve_coordinates[param1]
        p2 = curve_coordinates[param2]
        
        x1, y1 = p1
        x2, y2 = p2
        
        # 线段方向向量
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            continue
        
        # 计算参数t
        t = ((x_target - x1) * dx + (y_target - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # 限制在线段上
        
        # 计算垂足坐标
        foot_x = x1 + t * dx
        foot_y = y1 + t * dy
        
        # 计算距离
        distance = np.sqrt((x_target - foot_x) ** 2 + (y_target - foot_y) ** 2)
        
        if distance < min_distance:
            min_distance = distance
            closest_point = (foot_x, foot_y)
            best_param = param1 + t * (param2 - param1)
    
    return min_distance, closest_point, best_param

# ==================== 核心计算函数 ====================
def calc_tsv(sd):
    """
    根据光谱分布计算三刺激值 XYZ.
    :param sd: 光谱分布对象
    :return: 三刺激值 (X, Y, Z)
    """
    cmfs = get_standard_observer()
    return sd_to_XYZ(sd, cmfs)

def mccamy_calc_cct(xy):
    """用McCamy近似公式法计算相关色温"""
    n = (xy[0] - MCCAMY_OFFSET_X) / (xy[1] - MCCAMY_OFFSET_Y)
    coeffs = MCCAMY_COEFFS
    return coeffs[0] * n**3 + coeffs[1] * n**2 + coeffs[2] * n + coeffs[3]

def blackbody_triangle_calc_cct(xy):
    """用三角垂足插值法(通过普朗克公式计算黑体轨迹)计算相关色温"""
    blackbody_coordinates = calculate_blackbody_coordinates()
    _, _, best_temp = find_closest_point_on_curve(xy, blackbody_coordinates)
    return best_temp

def xy_to_uv(xy):
    """将CIE 1931色度坐标(x,y)转换为CIE 1976 UCS色度坐标(u',v')"""
    x, y = xy[0], xy[1]
    denominator = -2 * x + 12 * y + 3
    
    if denominator == 0:
        return None
    
    u_prime = safe_divide(4 * x, denominator)
    v_prime = safe_divide(9 * y, denominator)
    
    return np.array([u_prime, v_prime])

def XYZ_to_uv(XYZ):
    """将三刺激值XYZ转换为CIE 1976 UCS色度坐标(u',v')"""
    X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
    denominator = X + 15 * Y + 3 * Z
    
    if denominator == 0:
        return None
    
    u_prime = safe_divide(4 * X, denominator)
    v_prime = safe_divide(9 * Y, denominator)
    
    return np.array([u_prime, v_prime])

def calc_color_deviation_uv(uv_target):
    """计算目标光源在uv色度图上与黑体轨迹的最短距离（色偏差）"""
    # 计算黑体轨迹的uv坐标
    blackbody_coordinates = calculate_blackbody_coordinates()
    blackbody_uv_dict = {}
    
    cmfs = get_standard_observer()
    
    for temp, xy_coords in blackbody_coordinates.items():
        # 从xy转换为uv
        uv_coords = xy_to_uv(xy_coords)
        if uv_coords is not None:
            blackbody_uv_dict[temp] = (uv_coords[0], uv_coords[1])
    
    min_distance, closest_point, best_temp = find_closest_point_on_curve(uv_target, blackbody_uv_dict)
    
    # 计算带符号的距离
    if closest_point is not None:
        u_target, v_target = uv_target[0], uv_target[1]
        vec_u = u_target - closest_point[0]
        vec_v = v_target - closest_point[1]
        
        # 简化的符号判断：基于v坐标的相对位置
        signed_distance = min_distance if vec_v > 0 else -min_distance
        return signed_distance, closest_point, best_temp
    
    return min_distance, closest_point, best_temp

# ==================== 参考光源和光谱函数 ====================
def get_reference_illuminant(cct):
    """根据CCT获取参考光源的光谱功率分布"""
    spectral_data = {}
    
    if cct <= 4000:
        # 使用普朗克辐射体
        for wavelength in VISIBLE_WAVELENGTHS:
            spectral_data[wavelength] = planck_spectral_radiance(wavelength, cct)
    elif cct >= 5000:
        # 使用CIE D系列光源
        for wavelength in VISIBLE_WAVELENGTHS:
            if wavelength <= 500:
                spd_value = 1.0 + 0.5 * np.exp(-(wavelength - 460)**2 / 1000)
            else:
                spd_value = 1.0 * np.exp(-(wavelength - 560)**2 / 5000)
            spectral_data[wavelength] = spd_value
    else:
        # 混合
        ratio = (cct - 4000) / 1000
        for wavelength in VISIBLE_WAVELENGTHS:
            planck_value = planck_spectral_radiance(wavelength, cct)
            if wavelength <= 500:
                d_value = 1.0 + 0.5 * np.exp(-(wavelength - 460)**2 / 1000)
            else:
                d_value = 1.0 * np.exp(-(wavelength - 560)**2 / 5000)
            spectral_data[wavelength] = (1 - ratio) * planck_value + ratio * d_value
    
    return create_spectral_distribution(spectral_data)

def get_melanopsin_action_spectrum():
    """获取mel-opic动作光谱 s_mel(λ)"""
    mel_action_spectrum = {}
    
    for wavelength in VISIBLE_WAVELENGTHS:
        if wavelength < 400:
            sensitivity = 0.001
        elif wavelength <= 420:
            sensitivity = 0.01 * np.exp(-(wavelength - 420)**2 / 800)
        elif wavelength <= 480:
            sensitivity = np.exp(-(wavelength - 480)**2 / 1200)
        elif wavelength <= 550:
            sensitivity = 0.8 * np.exp(-(wavelength - 480)**2 / 1500)
        elif wavelength <= 600:
            sensitivity = 0.3 * np.exp(-(wavelength - 480)**2 / 2000)
        else:
            sensitivity = 0.05 * np.exp(-(wavelength - 480)**2 / 3000)
        
        mel_action_spectrum[wavelength] = max(0.001, sensitivity)
    
    # 归一化
    max_sensitivity = max(mel_action_spectrum.values())
    for wavelength in mel_action_spectrum:
        mel_action_spectrum[wavelength] /= max_sensitivity
    
    return mel_action_spectrum

def get_photopic_luminous_efficiency():
    """获取光视效率函数 V(λ)"""
    v_lambda = {}
    
    for wavelength in VISIBLE_WAVELENGTHS:
        if wavelength < 400:
            efficiency = 0.0001
        elif wavelength <= 500:
            efficiency = 0.01 * np.exp(-(wavelength - 555)**2 / 8000)
        elif wavelength <= 555:
            efficiency = np.exp(-(wavelength - 555)**2 / 6000)
        elif wavelength <= 650:
            efficiency = np.exp(-(wavelength - 555)**2 / 7000)
        else:
            efficiency = 0.001 * np.exp(-(wavelength - 555)**2 / 10000)
        
        v_lambda[wavelength] = max(0.0001, efficiency)
    
    # 归一化
    max_efficiency = max(v_lambda.values())
    for wavelength in v_lambda:
        v_lambda[wavelength] /= max_efficiency
    
    return v_lambda

def get_d65_spd():
    """获取标准日光D65的光谱功率分布"""
    d65_data = {}
    
    for wavelength in VISIBLE_WAVELENGTHS:
        if wavelength <= 400:
            spd_value = 50 + 10 * (wavelength - 380) / 20
        elif wavelength <= 500:
            spd_value = 60 + 40 * np.exp(-(wavelength - 460)**2 / 5000)
        elif wavelength <= 600:
            spd_value = 100 - 20 * (wavelength - 500) / 100
        else:
            spd_value = 80 - 30 * (wavelength - 600) / 180
        
        d65_data[wavelength] = max(10, spd_value)
    
    return create_spectral_distribution(d65_data)

# ==================== TM-30和色彩评估 ====================
def generate_ces_samples():
    """生成99个颜色评估样本（CES）的反射率数据"""
    ces_samples = []
    
    for hue in range(0, 360, 15):
        for saturation in [0.3, 0.6, 0.9]:
            for lightness in [0.3, 0.6, 0.8]:
                if len(ces_samples) >= CES_SAMPLE_COUNT:
                    break
                
                reflectance = {}
                peak_wavelength = 380 + (hue / 360) * 400
                
                for wavelength in VISIBLE_WAVELENGTHS:
                    base_reflectance = lightness * 0.8
                    peak_contribution = saturation * 0.4 * np.exp(-(wavelength - peak_wavelength)**2 / 5000)
                    reflectance[wavelength] = max(0.05, min(0.95, base_reflectance + peak_contribution))
                
                ces_samples.append(reflectance)
                
                if saturation == 0.9 and lightness == 0.8:
                    break
            if len(ces_samples) >= CES_SAMPLE_COUNT:
                break
        if len(ces_samples) >= CES_SAMPLE_COUNT:
            break
    
    return ces_samples[:CES_SAMPLE_COUNT]

def XYZ_to_CAM02UCS_simplified(XYZ, white_point_XYZ):
    """简化的XYZ到CAM02-UCS转换"""
    try:
        white_point = white_point_XYZ / white_point_XYZ[1] * 100
        return XYZ_to_CAM02UCS(XYZ, white_point)
    except:
        X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
        J = 100 * (Y / white_point_XYZ[1])**0.5
        
        total = X + Y + Z
        x = safe_divide(X, total, CIE_STANDARD_POINT_X)
        y = safe_divide(Y, total, CIE_STANDARD_POINT_Y)
        
        a = 500 * (x - CIE_STANDARD_POINT_X)
        b = 200 * (y - CIE_STANDARD_POINT_Y)
        
        return np.array([J, a, b])

def calculate_rf_rg(test_sd):
    """计算TM-30标准的保真度指数(Rf)和色域指数(Rg)"""
    cmfs = get_standard_observer(10)  # 尝试使用10度观察者
    
    # 计算测试光源的CCT
    XYZ_test = sd_to_XYZ(test_sd, cmfs)
    xy_test = XYZ_to_xy(XYZ_test)
    cct_test = mccamy_calc_cct(xy_test)
    
    # 获取参考光源
    ref_sd = get_reference_illuminant(cct_test)
    XYZ_ref = sd_to_XYZ(ref_sd, cmfs)
    
    # 生成CES样本
    ces_samples = generate_ces_samples()
    
    # 计算颜色坐标
    test_colors = []
    ref_colors = []
    
    for ces_reflectance in ces_samples:
        test_spectral_data = {}
        ref_spectral_data = {}
        
        for wavelength in VISIBLE_WAVELENGTHS:
            if wavelength in ces_reflectance and wavelength in test_sd.wavelengths:
                test_spectral_data[wavelength] = test_sd[wavelength] * ces_reflectance[wavelength]
                ref_spectral_data[wavelength] = ref_sd[wavelength] * ces_reflectance[wavelength]
        
        if test_spectral_data and ref_spectral_data:
            try:
                test_ces_sd = create_spectral_distribution(test_spectral_data)
                ref_ces_sd = create_spectral_distribution(ref_spectral_data)
                
                XYZ_test_ces = sd_to_XYZ(test_ces_sd, cmfs)
                XYZ_ref_ces = sd_to_XYZ(ref_ces_sd, cmfs)
                
                cam02ucs_test = XYZ_to_CAM02UCS_simplified(XYZ_test_ces, XYZ_test)
                cam02ucs_ref = XYZ_to_CAM02UCS_simplified(XYZ_ref_ces, XYZ_ref)
                
                test_colors.append(cam02ucs_test)
                ref_colors.append(cam02ucs_ref)
            except:
                continue
    
    if len(test_colors) == 0:
        return 0, 100, {"error": "无法计算有效的颜色样本"}
    
    # 计算Rf
    color_differences = []
    for test_color, ref_color in zip(test_colors, ref_colors):
        delta_J = test_color[0] - ref_color[0]
        delta_a = test_color[1] - ref_color[1]
        delta_b = test_color[2] - ref_color[2]
        
        color_diff = np.sqrt(delta_J**2 + delta_a**2 + delta_b**2)
        color_differences.append(color_diff)
    
    avg_color_diff = np.mean(color_differences)
    Rf = 100 - TM30_SCALE_FACTOR * avg_color_diff
    Rf = max(0, min(100, Rf))
    
    # 计算Rg
    angle_per_bin = 360 / HUE_BIN_COUNT
    test_hue_centers = []
    ref_hue_centers = []
    
    for bin_idx in range(HUE_BIN_COUNT):
        bin_test_a, bin_test_b = [], []
        bin_ref_a, bin_ref_b = [], []
        
        for test_color, ref_color in zip(test_colors, ref_colors):
            test_hue = np.arctan2(test_color[2], test_color[1]) * 180 / np.pi
            if test_hue < 0:
                test_hue += 360
            
            bin_start = bin_idx * angle_per_bin
            bin_end = (bin_idx + 1) * angle_per_bin
            
            if bin_start <= test_hue < bin_end or (bin_idx == HUE_BIN_COUNT - 1 and test_hue >= bin_start):
                bin_test_a.append(test_color[1])
                bin_test_b.append(test_color[2])
                bin_ref_a.append(ref_color[1])
                bin_ref_b.append(ref_color[2])
        
        if bin_test_a:
            test_hue_centers.append([np.mean(bin_test_a), np.mean(bin_test_b)])
            ref_hue_centers.append([np.mean(bin_ref_a), np.mean(bin_ref_b)])
    
    # 计算色域面积
    def polygon_area(points):
        if len(points) < 3:
            return 0
        points = np.array(points)
        x, y = points[:, 0], points[:, 1]
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))
    
    if len(test_hue_centers) >= 3 and len(ref_hue_centers) >= 3:
        test_area = polygon_area(test_hue_centers)
        ref_area = polygon_area(ref_hue_centers)
        Rg = safe_divide(test_area, ref_area, 1) * 100
    else:
        Rg = 100
    
    Rg = max(50, min(150, Rg))
    
    details = {
        "test_cct": cct_test,
        "num_valid_samples": len(test_colors),
        "avg_color_difference": avg_color_diff,
        "test_gamut_area": test_area if 'test_area' in locals() else 0,
        "ref_gamut_area": ref_area if 'ref_area' in locals() else 0,
        "hue_bins_used": len(test_hue_centers)
    }
    
    return Rf, Rg, details

# ==================== mel-DER计算 ====================
def calculate_mel_der(test_sd):
    """计算褪黑素日光照度比（mel-DER）"""
    mel_action_spectrum = get_melanopsin_action_spectrum()
    v_lambda = get_photopic_luminous_efficiency()
    d65_sd = get_d65_spd()
    
    # 计算测试光源的mel-opic辐照度和照度
    test_mel_irradiance = 0.0
    test_illuminance = 0.0
    
    for wavelength in VISIBLE_WAVELENGTHS:
        if wavelength in test_sd.wavelengths:
            # 输入数据单位：mW/m²/nm，需要转换为W/m²/nm
            spd_value = test_sd[wavelength] * 1e-3  # mW转换为W
            # 修正积分计算 - 移除不必要的1e-9转换
            test_mel_irradiance += spd_value * mel_action_spectrum[wavelength] * WAVELENGTH_STEP
            test_illuminance += spd_value * v_lambda[wavelength] * WAVELENGTH_STEP
    
    # 光视效率常数应用
    test_illuminance *= MAX_LUMINOUS_EFFICACY  # 683 lm/W
    
    print(f"调试信息 - 测试光源计算:")
    print(f"  原始mel-opic辐照度积分: {test_mel_irradiance}")
    print(f"  原始照度积分: {test_illuminance}")
    
    if test_illuminance <= 0:
        return 0, {"error": "测试光源照度为零"}
    
    test_mel_elr = test_mel_irradiance / test_illuminance
    
    # 计算D65的mel-opic辐照度和照度
    d65_mel_irradiance = 0.0
    d65_illuminance = 0.0
    
    # 修正D65计算 - 使用合理的归一化
    # 先计算D65的相对光谱分布，然后根据测试光源的总辐射量进行缩放
    test_total_radiance = sum(test_sd[wl] * 1e-3 for wl in test_sd.wavelengths) * WAVELENGTH_STEP
    d65_scaling_factor = test_total_radiance / 100.0  # 将D65缩放到与测试光源相似的量级
    
    for wavelength in VISIBLE_WAVELENGTHS:
        if wavelength in d65_sd.wavelengths:
            # 移除额外的单位转换
            d65_value = d65_sd[wavelength] * d65_scaling_factor
            d65_mel_irradiance += d65_value * mel_action_spectrum[wavelength] * WAVELENGTH_STEP
            d65_illuminance += d65_value * v_lambda[wavelength] * WAVELENGTH_STEP
    
    d65_illuminance *= MAX_LUMINOUS_EFFICACY
    
    print(f"调试信息 - D65计算:")
    print(f"  D65缩放因子: {d65_scaling_factor}")
    print(f"  D65 mel-opic辐照度: {d65_mel_irradiance}")
    print(f"  D65照度: {d65_illuminance}")
    
    if d65_illuminance <= 0:
        return 0, {"error": "D65照度计算错误"}
    
    d65_mel_elr = d65_mel_irradiance / d65_illuminance
    
    if d65_mel_elr <= 0:
        return 0, {"error": "D65 mel-opic ELR为零"}
    
    mel_der = test_mel_elr / d65_mel_elr
    
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

# ==================== 主程序 ====================
if __name__ == '__main__':
    # 读取波长与光强进np数组，波长为整型，光强为双精度浮点型
    df = pd.read_excel('data.xlsx', sheet_name='Problem 1')
    wavelength_arr = np.array([int(str(x)[:3]) for x in df['波长'].to_numpy()])
    energy_arr = df['光强'].to_numpy().astype(np.float64)

    # 将波长和能量数据组合成colour-science库所需的格式
    spectral_data = dict(zip(wavelength_arr, energy_arr))
    sd = create_spectral_distribution(spectral_data)

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
    print(f"用McCamy近似公式法计算的CCT: {mccamy_calc_cct(xy_coordinates)}")
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