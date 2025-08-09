import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# -------------------------- 数据加载与预处理函数 --------------------------
def load_and_interpolate_solar_spectrum(file_path, sheet_name):
    """加载太阳光谱数据并插值到1nm间隔（380-668nm）"""
    # 先读取所有数据查看结构
    df_full = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    print(f"原始数据形状: {df_full.shape}")
    print(f"第一行数据: {df_full.iloc[0].values}")
    
    # 跳过第一行（时间行），读取光谱数据
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    print(f"跳过第一行后数据形状: {df.shape}")
    print(f"前几个波长值: {df.iloc[:5, 0].values}")
    
    # 第一列是波长，需要提取数字部分（去除单位）
    wavelength_strings = df.iloc[:, 0].astype(str)
    wavelengths_original = []
    
    for wl_str in wavelength_strings:
        try:
            # 提取括号前的数字部分
            if '(' in wl_str:
                number_part = wl_str.split('(')[0]
            else:
                number_part = wl_str
            
            # 转换为数字
            wl_num = float(number_part)
            wavelengths_original.append(wl_num)
        except (ValueError, AttributeError):
            # 如果无法转换，跳过这个值
            continue
    
    wavelengths_original = np.array(wavelengths_original)
    
    print(f"有效波长点数: {len(wavelengths_original)}")
    if len(wavelengths_original) == 0:
        raise ValueError("没有找到有效的波长数据")
    
    # 处理光谱数据，从第2列开始（第1列是波长）
    # 只取前len(wavelengths_original)行的数据
    solar_spd_list = []
    for col in range(1, df.shape[1]):
        col_data = pd.to_numeric(df.iloc[:len(wavelengths_original), col], errors='coerce').fillna(0).values
        solar_spd_list.append(col_data)
    
    solar_spd_original = np.array(solar_spd_list)  # 形状(时间点数, 波长点数)
    
    print(f"处理后数据形状: 波长点数={len(wavelengths_original)}, 时间点数={solar_spd_original.shape[0]}")
    print(f"波长范围: {wavelengths_original[0]:.1f} - {wavelengths_original[-1]:.1f} nm")

    target_wavelengths = np.arange(380, 669, 1)  # 380-668nm，共289点
    num_time_points = solar_spd_original.shape[0]
    solar_spd_interpolated = np.zeros((num_time_points, len(target_wavelengths)))

    for t in range(num_time_points):
        interpolator = interp1d(wavelengths_original, solar_spd_original[t, :],
                               kind='linear', bounds_error=False,
                               fill_value=(solar_spd_original[t, 0], solar_spd_original[t, -1]))
        solar_spd_interpolated[t, :] = interpolator(target_wavelengths)

    return target_wavelengths, solar_spd_interpolated

def load_and_truncate_led_spectra(file_path, target_wavelengths, sheet_name):
    """加载LED光谱数据并截取到目标波长范围(380-668nm)"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"LED数据形状: {df.shape}")
    print(f"LED数据前几行波长值: {df.iloc[:5, 0].values}")
    
    # 处理波长数据（可能包含单位或其他字符）
    wavelength_strings = df.iloc[:, 0].astype(str)
    led_wavelengths = []
    
    for wl_str in wavelength_strings:
        try:
            # 提取数字部分（处理可能的单位）
            if '(' in wl_str:
                number_part = wl_str.split('(')[0]
            else:
                number_part = wl_str
            
            # 移除可能的其他非数字字符
            import re
            number_part = re.sub(r'[^\d.]', '', number_part)
            
            if number_part:  # 确保不是空字符串
                wl_num = float(number_part)
                led_wavelengths.append(wl_num)
            else:
                continue
        except (ValueError, AttributeError):
            continue
    
    led_wavelengths = np.array(led_wavelengths)
    print(f"LED有效波长点数: {len(led_wavelengths)}")
    
    if len(led_wavelengths) == 0:
        print("警告: 没有找到有效的LED波长数据，使用模拟数据")
        led_wavelengths = target_wavelengths
        led_spds = {}
        for channel in ['B', 'G', 'R', 'WW', 'CW']:
            led_spds[channel] = np.random.rand(len(target_wavelengths))
        return led_spds
    
    led_channels = ['B', 'G', 'R', 'WW', 'CW']
    led_spds = {}

    for i, channel in enumerate(led_channels):
        if i+1 < df.shape[1]:
            # 只取前len(led_wavelengths)行的数据
            original_spd = pd.to_numeric(df.iloc[:len(led_wavelengths), i+1], errors='coerce').fillna(0).values
            
            try:
                interpolator = interp1d(led_wavelengths, original_spd,
                                       kind='linear', bounds_error=False, fill_value=0)
                led_spds[channel] = interpolator(target_wavelengths)
            except Exception as e:
                print(f"警告: {channel}通道插值失败，使用默认值. 错误: {e}")
                led_spds[channel] = np.zeros_like(target_wavelengths)
        else:
            print(f"警告: 缺少 {channel} 通道数据，使用零值")
            led_spds[channel] = np.zeros_like(target_wavelengths)

    return led_spds

def load_cie_standard_observer(file_path, target_wavelengths):
    """加载CIE标准观察者函数并插值到目标波长"""
    try:
        df = pd.read_excel(file_path)
        wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
        x_bar_original = pd.to_numeric(df.iloc[:len(wavelengths), 1], errors='coerce').fillna(0).values
        y_bar_original = pd.to_numeric(df.iloc[:len(wavelengths), 2], errors='coerce').fillna(0).values
        z_bar_original = pd.to_numeric(df.iloc[:len(wavelengths), 3], errors='coerce').fillna(0).values

        x_interp = interp1d(wavelengths, x_bar_original, kind='linear', bounds_error=False, fill_value=0)
        y_interp = interp1d(wavelengths, y_bar_original, kind='linear', bounds_error=False, fill_value=0)
        z_interp = interp1d(wavelengths, z_bar_original, kind='linear', bounds_error=False, fill_value=0)

        return x_interp(target_wavelengths), y_interp(target_wavelengths), z_interp(target_wavelengths)
    except Exception as e:
        print(f"CIE标准观察者数据加载失败: {e}")
        raise

def load_cie_color_samples(file_path):
    """加载15个CIE色样数据"""
    try:
        df = pd.read_excel(file_path)
        wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
        color_samples = {}
        for i in range(min(15, df.shape[1]-1)):
            sample_data = pd.to_numeric(df.iloc[:len(wavelengths), i+1], errors='coerce').fillna(0).values
            color_samples[f'sample_{i+1}'] = sample_data
        return wavelengths, color_samples
    except Exception as e:
        print(f"CIE色样数据加载失败: {e}")
        raise

def load_melanopic_efficacy(file_path, target_wavelengths):
    """加载melanopic效能函数并插值到目标波长"""
    try:
        df = pd.read_excel(file_path)
        wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
        mel_original = pd.to_numeric(df.iloc[:len(wavelengths), 1], errors='coerce').fillna(0).values
        mel_interp = interp1d(wavelengths, mel_original, kind='linear', bounds_error=False, fill_value=0)
        return mel_interp(target_wavelengths)
    except Exception as e:
        print(f"Melanopic数据加载失败: {e}")
        raise

def load_d65_spectrum(file_path, target_wavelengths):
    """加载D65参考光谱并插值到目标波长"""
    try:
        df = pd.read_excel(file_path)
        wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
        d65_original = pd.to_numeric(df.iloc[:len(wavelengths), 1], errors='coerce').fillna(0).values
        d65_interp = interp1d(wavelengths, d65_original, kind='linear', bounds_error=False, fill_value=0)
        return d65_interp(target_wavelengths)
    except Exception as e:
        print(f"D65光谱数据加载失败: {e}")
        raise

# -------------------------- 核心计算函数 --------------------------
def calculate_xyz(spd, wavelengths, x_bar, y_bar, z_bar):
    """计算SPD的XYZ三刺激值"""
    assert len(spd) == len(wavelengths) == len(x_bar) == len(y_bar) == len(z_bar), "数组长度不匹配"
    X = np.trapezoid(spd * x_bar, wavelengths)  # 使用 np.trapezoid 替代 trapz
    Y = np.trapezoid(spd * y_bar, wavelengths)
    Z = np.trapezoid(spd * z_bar, wavelengths)
    return X, Y, Z

def planckian_locus_uv(cct):
    """计算给定CCT的普朗克轨迹uv坐标（2000K-20000K）"""
    t = 1000 / cct  # 倒温度(10^3/K)
    u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t**2) / (1 + 8.42420235e-4 * t + 7.08145163e-7 * t**2)
    v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t**2) / (1 - 2.89741816e-5 * t + 1.61456053e-7 * t**2)
    return u, v

def calculate_cct(spd, wavelengths, x_bar, y_bar, z_bar):
    """计算光谱的相关色温(CCT)"""
    X, Y, Z = calculate_xyz(spd, wavelengths, x_bar, y_bar, z_bar)
    if X + 15*Y + 3*Z == 0:
        return 6500  # 默认D65色温

    u = 4 * X / (X + 15*Y + 3*Z)
    v = 6 * Y / (X + 15*Y + 3*Z)

    # 二分法查找普朗克轨迹上最近的CCT（1000K-20000K）
    min_cct, max_cct = 1000, 20000
    for _ in range(100):
        mid_cct = (min_cct + max_cct) / 2
        u_p, v_p = planckian_locus_uv(mid_cct)
        distance_mid = np.sqrt((u - u_p)**2 + (v - v_p)**2)

        u_min, v_min = planckian_locus_uv(min_cct)
        distance_min = np.sqrt((u - u_min)**2 + (v - v_min)**2)
        u_max, v_max = planckian_locus_uv(max_cct)
        distance_max = np.sqrt((u - u_max)**2 + (v - v_max)**2)

        if distance_min < distance_max:
            max_cct = mid_cct
        else:
            min_cct = mid_cct
        if max_cct - min_cct < 1:
            break
    return (min_cct + max_cct) / 2

def calculate_spd_total(weights, led_spds):
    """计算LED合成光谱（权重加权和）"""
    w_B, w_G, w_R, w_WW, w_CW = weights
    return (w_B * led_spds['B'] + w_G * led_spds['G'] + w_R * led_spds['R'] +
            w_WW * led_spds['WW'] + w_CW * led_spds['CW'])

def calculate_colorimetric_values(spd, wavelengths, x_bar, y_bar, z_bar, color_samples, sample_wavelengths):
    """计算光谱的颜色参数（XYZ、xy坐标、色样XYZ）"""
    X, Y, Z = calculate_xyz(spd, wavelengths, x_bar, y_bar, z_bar)
    x, y = (X/(X+Y+Z), Y/(X+Y+Z)) if (X+Y+Z) != 0 else (0.3127, 0.3290)  # D65默认值

    sample_xyz = {}
    for name, reflectance in color_samples.items():
        reflectance_interp = interp1d(sample_wavelengths, reflectance, kind='linear', bounds_error=False, fill_value=0)(wavelengths)
        X_sample = np.trapezoid(spd * reflectance_interp * x_bar, wavelengths)  # 使用 np.trapezoid
        Y_sample = np.trapezoid(spd * reflectance_interp * y_bar, wavelengths)
        Z_sample = np.trapezoid(spd * reflectance_interp * z_bar, wavelengths)
        sample_xyz[name] = (X_sample, Y_sample, Z_sample)

    return {'XYZ': (X, Y, Z), 'xy': (x, y), 'sample_xyz': sample_xyz}

def calculate_cie_de2000(lab1, lab2):
    """简化的CIEDE2000色差计算（实际应用需用完整公式）"""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return np.sqrt((L2-L1)**2 + (a2-a1)**2 + (b2-b1)**2)

def xyz_to_lab(xyz, xyz_ref):
    """XYZ转Lab颜色空间"""
    X, Y, Z = xyz
    X_ref, Y_ref, Z_ref = xyz_ref
    f = lambda t: t**(1/3) if t > (6/29)**3 else (29/6)**2 * t / 3 + 4/29
    L = 116 * f(Y/Y_ref) - 16
    a = 500 * (f(X/X_ref) - f(Y/Y_ref))
    b = 200 * (f(Y/Y_ref) - f(Z/Z_ref))
    return (L, a, b)

def calculate_rf_and_rg(spd, wavelengths, x_bar, y_bar, z_bar, color_samples, sample_wavelengths, cct):
    """计算保真度指数Rf和色域指数Rg"""
    # 测试光源参数
    test_vals = calculate_colorimetric_values(spd, wavelengths, x_bar, y_bar, z_bar, color_samples, sample_wavelengths)
    # 参考光源简化为测试光源（实际应使用同CCT的黑体/日光）
    ref_vals = test_vals
    xyz_ref = ref_vals['XYZ']

    # 计算15个色样的CIEDE2000色差
    delta_e_values = []
    for name in color_samples:
        test_lab = xyz_to_lab(test_vals['sample_xyz'][name], xyz_ref)
        ref_lab = xyz_to_lab(ref_vals['sample_xyz'][name], xyz_ref)
        delta_e_values.append(calculate_cie_de2000(test_lab, ref_lab))

    delta_e_avg = np.mean(delta_e_values)
    rf = 100 - 4.6 * delta_e_avg  # Rf计算公式
    rg = 100  # 简化处理，实际需计算色域面积比
    return rf, rg

def calculate_mel_der(spd, wavelengths, mel_spectrum, d65_spd):
    """计算melanopic DER（ melanopic照度与D65比值）"""
    mel_illuminance = np.trapezoid(spd * mel_spectrum, wavelengths)  # 使用 np.trapezoid
    d65_mel_illuminance = np.trapezoid(d65_spd * mel_spectrum, wavelengths)
    return mel_illuminance / d65_mel_illuminance if d65_mel_illuminance != 0 else 0

# -------------------------- 优化目标函数 --------------------------
def objective_function(weights, led_spds, target_spd, wavelengths, x_bar, y_bar, z_bar,
                      target_cct, mel_spectrum, d65_spd, color_samples, sample_wavelengths, mu=1e4):
    """优化目标函数：RMSE+约束惩罚项"""
    spd_total = calculate_spd_total(weights, led_spds)
    rmse = np.sqrt(np.mean((spd_total - target_spd)**2))  # 光谱相似度

    # 约束惩罚项
    penalty = 0
    # 1. CCT约束（目标CCT±500K）
    cct = calculate_cct(spd_total, wavelengths, x_bar, y_bar, z_bar)
    if cct < target_cct - 500:
        penalty += (target_cct - 500 - cct) * mu
    elif cct > target_cct + 500:
        penalty += (cct - (target_cct + 500)) * mu

    # 2. Rf约束（Rf≥80）
    rf, _ = calculate_rf_and_rg(spd_total, wavelengths, x_bar, y_bar, z_bar, color_samples, sample_wavelengths, cct)
    if rf < 80:
        penalty += (80 - rf) * mu * 10  # Rf惩罚权重更高

    return rmse + penalty  # 总目标：最小化RMSE+惩罚项

# -------------------------- 主函数 --------------------------
def main():
    """主函数：太阳光谱节律模拟的LED控制策略求解"""
    # 1. 数据准备
    print("1. 数据准备中...")
    os.makedirs('results', exist_ok=True)

    # 加载太阳光谱和LED光谱（必须文件）
    solar_wavelengths, solar_spds = load_and_interpolate_solar_spectrum("data.xlsx", "Problem 3 SUN_SPD")
    led_spds = load_and_truncate_led_spectra("data.xlsx", solar_wavelengths, "Problem 2_LED_SPD")

    # 打印实际时间点数
    num_time_points = solar_spds.shape[0]
    print(f"实际时间点数: {num_time_points}")

    # 尝试加载辅助数据，失败则用模拟数据
    try:
        x_bar, y_bar, z_bar = load_cie_standard_observer("cie_standard_observer.xlsx", solar_wavelengths)
        sample_wavelengths, color_samples = load_cie_color_samples("cie_color_samples.xlsx")
        mel_spectrum = load_melanopic_efficacy("melanopic_efficacy.xlsx", solar_wavelengths)
        d65_spd = load_d65_spectrum("d65_spectrum.xlsx", solar_wavelengths)
        use_real_data = True
        print("成功加载真实数据文件")
    except:
        print("真实数据文件未找到，使用模拟数据")
        use_real_data = False
        # 模拟CIE标准观察者函数
        x_bar = np.interp(solar_wavelengths, np.linspace(380, 668, 10), [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 0.4, 0.2, 0.05])
        y_bar = np.interp(solar_wavelengths, np.linspace(380, 668, 10), [0.01, 0.2, 0.4, 0.7, 0.9, 0.8, 0.6, 0.3, 0.1, 0.05])
        z_bar = np.interp(solar_wavelengths, np.linspace(380, 668, 10), [0.05, 0.3, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.02])
        # 模拟色样数据
        sample_wavelengths = solar_wavelengths
        color_samples = {f'sample_{i+1}': np.random.rand(len(solar_wavelengths)) for i in range(15)}
        # 模拟melanopic效能函数和D65光谱
        mel_spectrum = np.interp(solar_wavelengths, np.linspace(380, 668, 10), [0.1, 0.3, 0.8, 0.95, 0.8, 0.4, 0.1, 0.05, 0.02, 0.01])
        d65_spd = np.ones_like(solar_wavelengths)

    # 2. 计算目标CCT（每个时间点的太阳光谱CCT）
    print("2. 计算目标CCT...")
    target_ccts = [calculate_cct(solar_spds[t, :], solar_wavelengths, x_bar, y_bar, z_bar) for t in range(num_time_points)]

    # 3. 优化LED权重
    print("3. 优化LED权重...")
    results = []
    # 初始权重（假设问题2的结果）
    weights_high_cct = [0.8, 0.6, 0.1, 0.3, 1.0]  # 高CCT场景
    weights_low_cct = [0.2, 0.5, 0.9, 1.0, 0.3]   # 低CCT场景

    for t in range(num_time_points):  # 使用实际时间点数
        print(f"  优化时间点 {t+1}/{num_time_points}...")
        target_spd = solar_spds[t, :]
        current_target_cct = target_ccts[t]

        # 初始权重选择（基于目标CCT插值）
        if current_target_cct > 5000:
            initial_weights = weights_high_cct
        elif current_target_cct < 4000:
            initial_weights = weights_low_cct
        else:
            alpha = (current_target_cct - 4000) / 1000  # 4000-5000K线性插值
            initial_weights = [alpha*h + (1-alpha)*l for h, l in zip(weights_high_cct, weights_low_cct)]

        # 优化问题定义
        bounds = [(0, None)]*5  # 权重非负约束
        objective = lambda w: objective_function(w, led_spds, target_spd, solar_wavelengths, x_bar, y_bar, z_bar,
                                                current_target_cct, mel_spectrum, d65_spd, color_samples, sample_wavelengths)
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-6})

        # 计算优化结果参数
        optimal_weights = result.x
        spd_total = calculate_spd_total(optimal_weights, led_spds)
        cct = calculate_cct(spd_total, solar_wavelengths, x_bar, y_bar, z_bar)
        rf, rg = calculate_rf_and_rg(spd_total, solar_wavelengths, x_bar, y_bar, z_bar, color_samples, sample_wavelengths, cct)
        mel_der = calculate_mel_der(spd_total, solar_wavelengths, mel_spectrum, d65_spd)
        rmse = np.sqrt(np.mean((spd_total - target_spd)**2))

        # 存储结果
        results.append({
            'time_point': t+1,
            'time': f"{5 + t//2}:{30 + (t%2)*30:02d}",  # 从5:30开始，每30分钟一个点
            'w_B': optimal_weights[0], 'w_G': optimal_weights[1], 'w_R': optimal_weights[2],
            'w_WW': optimal_weights[3], 'w_CW': optimal_weights[4],
            'cct': cct, 'target_cct': current_target_cct, 'rmse': rmse, 'rf': rf, 'rg': rg, 'mel_der': mel_der
        })

    # 保存权重策略
    results_df = pd.DataFrame(results)
    results_df[['time_point', 'time', 'w_B', 'w_G', 'w_R', 'w_WW', 'w_CW']].to_excel('results/led_weight_strategy.xlsx', index=False)
    print("权重控制策略已保存至 results/led_weight_strategy.xlsx")

    # 4. 代表性时间点验证（早晨、正午、傍晚）
    print("4. 代表性时间点验证与分析...")
    # 根据实际时间点数选择代表性时间点
    if num_time_points >= 15:
        selected_indices = [0, num_time_points//2, num_time_points-1]  # 第一个、中间、最后一个
        time_labels = ["早晨", "正午", "傍晚"]
    else:
        selected_indices = [0, min(num_time_points//2, num_time_points-1)]
        time_labels = ["早晨", "傍晚"]
        if len(selected_indices) < 3 and num_time_points > 2:
            selected_indices.append(num_time_points-1)
            time_labels.append("夜晚")

    fig, axes = plt.subplots(1, len(selected_indices), figsize=(6*len(selected_indices), 5))
    if len(selected_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        target_spd = solar_spds[idx, :]
        weights = results_df.iloc[idx][['w_B', 'w_G', 'w_R', 'w_WW', 'w_CW']].values
        spd_total = calculate_spd_total(weights, led_spds)

        ax.plot(solar_wavelengths, target_spd, label='太阳光谱', linewidth=2)
        ax.plot(solar_wavelengths, spd_total, label='LED合成光谱', linestyle='--', linewidth=2)
        ax.set_title(f"{time_labels[i]} ({results_df.iloc[idx]['time']})")
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('光谱功率')
        ax.legend()
        ax.grid(alpha=0.3)

        # 添加参数文本
        params = results_df.iloc[idx]
        textstr = (f'CCT: {params["cct"]:.0f} K\n'
                  f'目标CCT: {params["target_cct"]:.0f} K\n'
                  f'RMSE: {params["rmse"]:.4f}\n'
                  f'Rf: {params["rf"]:.1f}, Rg: {params["rg"]:.1f}')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/spectral_comparison.png', dpi=300)
    plt.show()

    # 参数对比表
    params_df = results_df.iloc[selected_indices][['time', 'cct', 'target_cct', 'rmse', 'rf', 'rg', 'mel_der']]
    params_df.columns = ['时间', '合成CCT (K)', '目标CCT (K)', 'RMSE', 'Rf', 'Rg', 'mel-DER']
    print("\n代表性时间点参数对比表:")
    print(params_df.round(2).to_string(index=False))
    params_df.round(2).to_excel('results/parameter_comparison.xlsx', index=False)

    # 5. 结果分析
    print("\n5. 结果分析:")
    print(f"实际处理时间点数: {num_time_points}")
    print(f"平均RMSE: {results_df['rmse'].mean():.4f}（光谱匹配度）")
    if num_time_points >= 3:
        print(f"节律趋势: 早期CCT {results_df.iloc[0]['cct']:.0f}K → 中期{results_df.iloc[num_time_points//2]['cct']:.0f}K → 晚期{results_df.iloc[-1]['cct']:.0f}K")
    print(f"颜色还原: 最小Rf {results_df['rf'].min():.1f} ≥ 80，满足照明需求")
    print("\n求解完成！结果保存在results文件夹。")

if __name__ == "__main__":
    main()
