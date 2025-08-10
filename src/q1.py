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

# TODO: 修复 pandas 兼容性问题
# 具体步骤：
# 1. 找到 <anaconda root dir>\envs\<env_name>\Lib\site-packages\pyplr\CIE.py 文件
# 2. 定位 get_CIES026 函数
# 3. 将其中的代码行
#       sss.index = pd.Int64Index(sss.index)
#    替换为
#       sss.index = sss.index.astype('int64')
from pyplr.CIE import get_CIES026

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

# ==================== TM-30和色彩评估 ====================
def calc_rf_rg(spd_dict):
    """使用colour库的TM-30方法计算Rf和Rg"""
    spd = colour.SpectralDistribution(spd_dict, name="SPD")
    results = colour.colour_fidelity_index(
        spd, additional_data=True, method="ANSI/IES TM-30-18"
    )  # 不加additional_data=True只会返回Rf
    Rf = results.R_f
    Rg = results.R_g
    return Rf, Rg

# ==================== mel-DER计算 ====================
def calculate_mel_DER(light_source_sd):
    """计算 melanopic DER (mel-DER)"""
    cie_s026 = get_CIES026()
    melanopic_sensitivity_data = cie_s026["I"]  # melanopic 灵敏度数据
    mel_spd_dict = dict(
        zip(melanopic_sensitivity_data.index.values, melanopic_sensitivity_data.values)
    )
    melanopic_sensitivity = colour.SpectralDistribution(mel_spd_dict)

    D65 = colour.SDS_ILLUMINANTS["D65"]  # D65 光源

    min_wvl = max(
        melanopic_sensitivity.wavelengths.min(),
        light_source_sd.wavelengths.min(),
        D65.wavelengths.min(),
    )
    max_wvl = min(
        melanopic_sensitivity.wavelengths.max(),
        light_source_sd.wavelengths.max(),
        D65.wavelengths.max(),
    )
    wvls = colour.SpectralShape(min_wvl, max_wvl, 1)

    melanopic_sensitivity = melanopic_sensitivity.interpolate(wvls)
    light_source_sd = light_source_sd.interpolate(wvls)
    D65 = D65.interpolate(wvls)

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    cmfs = cmfs.copy().align(wvls)
    cmfs_v_dict = dict(zip(cmfs.wavelengths, cmfs.values[:, 1]))
    V_lambda = colour.SpectralDistribution(cmfs_v_dict)  # 明视觉函数数据

    # melanopic有效辐照度
    E_mel_test = np.trapezoid(
        light_source_sd.values * melanopic_sensitivity.values,
        light_source_sd.wavelengths,
    )

    # 明视觉照度
    E_v_test = np.trapezoid(
        light_source_sd.values * V_lambda.values,
        light_source_sd.wavelengths,
    )

    # D65光源的melanopic有效辐照度
    E_mel_D65 = np.trapezoid(
        D65.values * melanopic_sensitivity.values,
        D65.wavelengths,
    )

    # D65光源的明视觉照度
    E_v_D65 = np.trapezoid(
        D65.values * V_lambda.values,
        D65.wavelengths,
    )

    # 测试光源的melanopic与明视觉的比值(mel-ELR)
    K_mel_v_test = E_mel_test / E_v_test
    # D65光源的melanopic与明视觉的比值(mel-EDI)
    K_mel_v_D65 = E_mel_D65 / E_v_D65
    # 测试光源与D65光源的melanopic效应比值(mel-DER)
    mel_DER = K_mel_v_test / K_mel_v_D65

    # 打印中间计算参数
    print(f"测试光源 melanopic 有效辐照度 (E_mel_test): {E_mel_test:.6f}")
    print(f"测试光源明视觉照度 (E_v_test): {E_v_test:.6f}")
    print(f"测试光源 melanopic ELR (K_mel_v_test): {K_mel_v_test:.6f}")
    print(f"D65光源 melanopic 有效辐照度 (E_mel_D65): {E_mel_D65:.6f}")
    print(f"D65光源明视觉照度 (E_v_D65): {E_v_D65:.6f}")
    print(f"D65光源 melanopic ELR (K_mel_v_D65): {K_mel_v_D65:.6f}")

    return mel_DER

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
    Rf, Rg = calc_rf_rg(spectral_data)
    print(f"保真度指数 (Rf): {Rf}")
    print(f"色域指数 (Rg): {Rg}")
    
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
    mel_der = calculate_mel_DER(sd)
    print(f"褪黑素日光照度比 (mel-DER): {mel_der:.4f}")
    
    # mel-DER评估
    if mel_der > 1.0:
        mel_assessment = "相比D65日光，该光源对melanopsin的刺激更强"
    elif mel_der < 0.8:
        mel_assessment = "相比D65日光，该光源对melanopsin的刺激较弱，可能更适合晚间使用"
    else:
        mel_assessment = "该光源对melanopsin的刺激与D65日光相近"
    
    print(f"mel-DER评估: {mel_assessment}")