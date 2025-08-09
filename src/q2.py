# 多通道LED光源日间照明模式优化
# 基于遗传算法的光谱合成优化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# 从q1.py导入现成的计算函数
from q1 import (
    create_spectral_distribution,
    calc_tsv,
    mccamy_calc_cct,
    blackbody_triangle_calc_cct,
    XYZ_to_xy,
    xy_to_uv,
    XYZ_to_uv,
    calculate_rf_rg,
    calculate_mel_der,
    calc_color_deviation_uv
)

class MultiChannelLEDOptimizer:
    def __init__(self, data_file='data.xlsx'):
        """初始化优化器"""
        self.load_led_data(data_file)
        self.iteration_count = 0  # 添加迭代计数器
        
    def load_led_data(self, data_file):
        """加载LED SPD数据"""
        df = pd.read_excel(data_file, sheet_name='Problem 2_LED_SPD')
        
        # 提取波长数据（从字符串中提取数值）
        self.wavelengths = np.array([int(str(x)[:3]) for x in df['波长'].to_numpy()])
        
        # 提取各通道SPD数据
        self.spd_data = {
            'Red': df['Red'].to_numpy().astype(np.float64),
            'Green': df['Green'].to_numpy().astype(np.float64), 
            'Blue': df['Blue'].to_numpy().astype(np.float64),
            'Warm_White': df['Warm White'].to_numpy().astype(np.float64),
            'Cold_White': df['Cold White'].to_numpy().astype(np.float64)
        }
        
        # 归一化SPD数据
        for channel in self.spd_data:
            max_val = np.max(self.spd_data[channel])
            if max_val > 0:
                self.spd_data[channel] = self.spd_data[channel] / max_val
                
        print(f"已加载LED数据，波长范围：{self.wavelengths[0]}-{self.wavelengths[-1]}nm")
        
    def synthesize_spectrum(self, weights):
        """合成光谱"""
        w_red, w_green, w_blue, w_ww, w_cw = weights
        
        total_spd = (w_red * self.spd_data['Red'] + 
                    w_green * self.spd_data['Green'] +
                    w_blue * self.spd_data['Blue'] + 
                    w_ww * self.spd_data['Warm_White'] +
                    w_cw * self.spd_data['Cold_White'])
        
        # 归一化
        if np.sum(total_spd) > 0:
            total_spd = total_spd / np.sum(total_spd)
            
        return total_spd
        
    def calculate_optical_parameters(self, spd):
        """计算光学参数 - 增强错误处理和默认值"""
        # 检查光谱有效性
        if np.sum(spd) <= 0 or np.any(np.isnan(spd)) or np.any(np.isinf(spd)):
            print("警告：无效光谱数据")
            return {
                'XYZ': np.array([0, 0, 0]),
                'xy': np.array([0.3127, 0.3290]),  # D65白点
                'CCT': 6500,
                'Rf': 0,
                'Rg': 100,
                'mel_DER': 0.8
            }
    
        # 创建光谱分布对象
        spectral_data = dict(zip(self.wavelengths, spd))
        sd = create_spectral_distribution(spectral_data)
        
        # 计算三刺激值XYZ
        try:
            XYZ = calc_tsv(sd)
            if np.any(np.isnan(XYZ)) or np.any(np.isinf(XYZ)):
                raise ValueError("XYZ包含无效值")
        except Exception as e:
            print(f"XYZ计算失败: {e}")
            XYZ = np.array([95.047, 100.0, 108.883])  # D65标准光源XYZ
    
        # 计算色品坐标
        try:
            xy = XYZ_to_xy(XYZ)
            if np.any(np.isnan(xy)) or np.any(np.isinf(xy)):
                raise ValueError("xy包含无效值")
        except Exception as e:
            print(f"xy计算失败: {e}")
            xy = np.array([0.3127, 0.3290])
    
        # 计算CCT (使用McCamy公式)
        try:
            cct = mccamy_calc_cct(xy)
            if np.isnan(cct) or np.isinf(cct) or cct <= 0:
                # 尝试备用方法
                cct = blackbody_triangle_calc_cct(xy)
                if np.isnan(cct) or np.isinf(cct) or cct <= 0:
                    cct = 6500  # 使用默认值
        except Exception as e:
            print(f"CCT计算失败: {e}")
            cct = 6500
    
        # 计算TM-30指数 (Rf, Rg) - 增强错误处理
        rf, rg = 80, 100  # 默认值
        try:
            rf_calc, rg_calc, tm30_details = calculate_rf_rg(sd)
            if not (np.isnan(rf_calc) or np.isinf(rf_calc) or rf_calc < 0):
                rf = rf_calc
            if not (np.isnan(rg_calc) or np.isinf(rg_calc) or rg_calc < 0):
                rg = rg_calc
        except Exception as e:
            print(f"TM-30计算出错: {e}")
    
        # 计算mel-DER
        mel_der = 0.8  # 默认值
        try:
            mel_der_calc, mel_details = calculate_mel_der(sd)
            if not (np.isnan(mel_der_calc) or np.isinf(mel_der_calc) or mel_der_calc < 0):
                mel_der = mel_der_calc
        except Exception as e:
            print(f"mel-DER计算出错: {e}")
    
        return {
            'XYZ': XYZ,
            'xy': xy,
            'CCT': cct,
            'Rf': rf,
            'Rg': rg,
            'mel_DER': mel_der
        }
        
    def fitness_function(self, weights):
        """适应度函数（目标函数）"""
        # 权重约束：0 <= wi <= 1
        if np.any(weights < 0) or np.any(weights > 1):
            return -1000  # 惩罚
            
        # 合成光谱
        spd = self.synthesize_spectrum(weights)
        
        # 计算光学参数（使用q1.py中的函数）
        try:
            params = self.calculate_optical_parameters(spd)
            cct = params['CCT']
            rf = params['Rf']
            rg = params['Rg']
            mel_der = params['mel_DER']
        except Exception as e:
            print(f"参数计算出错: {e}")
            return -1000
        
        # 多目标优化：同时考虑Rf和Rg
        rf_score = rf / 100.0
        rg_score = 1.0 if 95 <= rg <= 105 else max(0, 1 - abs(rg - 100) / 20)
        
        # 综合目标函数
        objective = 0.6 * rf_score + 0.4 * rg_score
        
        # 约束惩罚
        penalty = 0
        penalty_weight = 5000  # 增加惩罚权重
        
        # CCT约束：6500K ± 500K
        if abs(cct - 6500) > 500:
            penalty += penalty_weight * (abs(cct - 6500) - 500) / 500
            
        # Rg约束：[95, 105] - 强化约束
        if rg < 95:
            penalty += penalty_weight * (95 - rg) / 5  # 增大惩罚
        elif rg > 105:
            penalty += penalty_weight * (rg - 105) / 5
            
        # Rf最低要求：> 88
        if rf < 88:
            penalty += penalty_weight * (88 - rf) / 12
            
        fitness = objective - penalty
        return -fitness  # 转换为最小化问题
        
    def test_callback(self):
        """测试回调函数是否工作"""
        print("正在测试回调函数...")
        
        def simple_func(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        def simple_callback(xk, convergence):
            print(f"回调测试: x={xk}, convergence={convergence}")
            return False
        
        bounds = [(0, 5), (0, 5)]
        try:
            result = differential_evolution(
                simple_func, 
                bounds, 
                maxiter=5,  # 只测试5次迭代
                popsize=10,
                callback=simple_callback
            )
            print(f"回调测试完成，结果: {result.x}")
            print("回调函数工作正常！")
        except Exception as e:
            print(f"回调测试失败: {e}")
            print("将使用内置监控功能替代回调")
    
    def fitness_function_with_monitoring(self, weights):
        """带监控的适应度函数"""
        # 增加调用计数器
        if not hasattr(self, 'eval_count'):
            self.eval_count = 0
        self.eval_count += 1
        
        # 每20次评估输出一次进度信息
        if self.eval_count % 20 == 0:
            print(f"评估进度: {self.eval_count}/50000 ({self.eval_count/500:.1f}%)")
        
        # 权重约束：0 <= wi <= 1
        if np.any(weights < 0) or np.any(weights > 1):
            return -1000
            
        # 合成光谱
        spd = self.synthesize_spectrum(weights)
        
        # 计算光学参数
        try:
            params = self.calculate_optical_parameters(spd)
            cct = params['CCT']
            rf = params['Rf']
            rg = params['Rg']
            mel_der = params['mel_DER']
            
            # 每20次评估输出详细信息
            if self.eval_count % 20 == 0:
                print(f"\n=== 详细进度 - 评估 {self.eval_count} ===")
                print(f"当前权重: [{', '.join([f'{w:.3f}' for w in weights])}]")
                print(f"CCT: {cct:.1f}K, Rf: {rf:.1f}, Rg: {rg:.1f}, mel-DER: {mel_der:.3f}")
                
                # 约束检查
                constraints = []
                if 6000 <= cct <= 7000:
                    constraints.append("CCT✓")
                else:
                    constraints.append("CCT✗")
                if rf > 88:
                    constraints.append("Rf✓")
                else:
                    constraints.append("Rf✗")
                if 95 <= rg <= 105:
                    constraints.append("Rg✓")
                else:
                    constraints.append("Rg✗")
                print(f"约束满足: {' '.join(constraints)}")
                print("-" * 40)
                
        except Exception as e:
            if self.eval_count % 100 == 0:
                print(f"评估 {self.eval_count}: 参数计算出错 - {e}")
            return -1000
        
        # 多目标优化：同时考虑Rf和Rg
        rf_score = rf / 100.0
        rg_score = 1.0 if 95 <= rg <= 105 else max(0, 1 - abs(rg - 100) / 20)
        
        # 综合目标函数
        objective = 0.6 * rf_score + 0.4 * rg_score
        
        # 约束惩罚
        penalty = 0
        penalty_weight = 5000
        
        # CCT约束：6500K ± 500K
        if abs(cct - 6500) > 500:
            penalty += penalty_weight * (abs(cct - 6500) - 500) / 500
            
        # Rg约束：[95, 105]
        if rg < 95:
            penalty += penalty_weight * (95 - rg) / 5
        elif rg > 105:
            penalty += penalty_weight * (rg - 105) / 5
            
        # Rf最低要求：> 88
        if rf < 88:
            penalty += penalty_weight * (88 - rf) / 12
            
        fitness = objective - penalty
        return -fitness
    
    def optimization_callback(self, xk, convergence):
        """优化过程回调函数，每迭代20次输出一次数据"""
        self.iteration_count += 1  # 每次回调增加1
        
        if self.iteration_count % 20 == 0:  # 每20次迭代输出一次
            # 计算当前解的性能
            spd = self.synthesize_spectrum(xk)
            try:
                params = self.calculate_optical_parameters(spd)
                print(f"\n迭代 {self.iteration_count:4d}:")
                print(f"  当前权重: [{', '.join([f'{w:.3f}' for w in xk])}]")
                print(f"  CCT: {params['CCT']:.1f}K")
                print(f"  Rf:  {params['Rf']:.1f}")
                print(f"  Rg:  {params['Rg']:.1f}")
                print(f"  mel-DER: {params['mel_DER']:.3f}")
                print(f"  收敛度: {convergence:.6f}")
                
                # 检查约束满足情况
                constraints_met = []
                if 6000 <= params['CCT'] <= 7000:
                    constraints_met.append("CCT✓")
                else:
                    constraints_met.append("CCT✗")
                    
                if params['Rf'] > 88:
                    constraints_met.append("Rf✓")
                else:
                    constraints_met.append("Rf✗")
                    
                if 95 <= params['Rg'] <= 105:
                    constraints_met.append("Rg✓")
                else:
                    constraints_met.append("Rg✗")
                    
                print(f"  约束: {' '.join(constraints_met)}")
                
            except Exception as e:
                print(f"\n迭代 {self.iteration_count:4d}: 参数计算出错 - {e}")
            
        return False  # 继续优化
        
    def optimize_daytime_lighting(self):
        """优化日间照明模式"""
        print("开始优化日间照明模式...")
        print("目标：CCT = 6500K ± 500K，最大化Rf，Rg ∈ [95,105]")
        print("-" * 60)
        
        # 重置计数器
        self.iteration_count = 0
        self.eval_count = 0
        
        # 权重边界：[0, 1]
        bounds = [(0, 1) for _ in range(5)]
        
        # 先测试回调函数是否工作
        print("测试回调函数...")
        self.test_callback()
        
        print("\n开始实际优化...")
        print("预计总评估次数: ~50,000")
        print("将每20次评估输出一次进度信息")
        print("=" * 60)
        
        # 使用带监控的适应度函数
        result = differential_evolution(
            self.fitness_function_with_monitoring,  # 使用监控版本
            bounds,
            maxiter=1000,
            popsize=50,
            atol=1e-6,
            seed=42,
            disp=True,  # 显示优化过程
            callback=self.optimization_callback
        )
        
        print(f"\n" + "="*60)
        print(f"优化完成！")
        print(f"总评估次数: {getattr(self, 'eval_count', 0)}")
        if result.success:
            optimal_weights = result.x
            print(f"优化成功！总迭代次数: {result.nit}")
        else:
            optimal_weights = result.x
            print(f"优化完成（可能未完全收敛）。总迭代次数: {result.nit}")
            
        return optimal_weights

    def evaluate_solution(self, weights):
        """评估解的性能"""
        # 合成光谱
        spd = self.synthesize_spectrum(weights)
        
        # 计算所有参数（使用q1.py中的函数）
        params = self.calculate_optical_parameters(spd)
        
        return {
            'weights': weights,
            'CCT': params['CCT'],
            'Rf': params['Rf'],
            'Rg': params['Rg'],
            'mel_DER': params['mel_DER'],
            'XYZ': params['XYZ'],
            'xy': params['xy'],
            'spd': spd
        }
        
    def plot_results(self, results):
        """绘制结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制合成光谱
        ax1.plot(self.wavelengths, results['spd'], 'r-', linewidth=2, label='合成光谱')
        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('相对光谱功率')
        ax1.set_title('多通道LED合成光谱')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制各通道贡献
        weights = results['weights']
        channels = ['Red', 'Green', 'Blue', 'Warm_White', 'Cold_White']
        colors = ['red', 'green', 'blue', 'orange', 'cyan']
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            if weights[i] > 0.01:  # 只显示有意义的贡献
                ax2.plot(self.wavelengths, weights[i] * self.spd_data[channel], 
                        color=color, linewidth=1.5, alpha=0.7,
                        label=f'{channel}: {weights[i]:.3f}')
        
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('相对光谱功率')
        ax2.set_title('各通道光谱贡献')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """主函数"""
    # 创建优化器
    optimizer = MultiChannelLEDOptimizer('data.xlsx')
    
    # 优化日间照明模式
    optimal_weights = optimizer.optimize_daytime_lighting()
    
    # 评估最优解
    results = optimizer.evaluate_solution(optimal_weights)
    
    # 输出结果
    print("\n" + "="*50)
    print("多通道LED光源日间照明模式优化结果")
    print("="*50)
    print(f"最优权重组合:")
    channels = ['深红光', '绿光', '蓝光', '暖白光', '冷白光']
    for i, (channel, weight) in enumerate(zip(channels, optimal_weights)):
        print(f"  {channel}: {weight:.4f}")
    
    print(f"\n光学性能参数:")
    print(f"  相关色温 (CCT): {results['CCT']:.1f} K")
    print(f"  保真度指数 (Rf): {results['Rf']:.1f}")
    print(f"  色域指数 (Rg): {results['Rg']:.1f}")
    print(f"  视黑素日光效率比 (mel-DER): {results['mel_DER']:.3f}")
    print(f"  三刺激值 (XYZ): {results['XYZ']}")
    print(f"  色品坐标 (xy): {results['xy']}")
    
    # 使用q1.py的函数计算色偏差
    try:
        uv_coords = XYZ_to_uv(results['XYZ'])
        color_deviation, closest_point, closest_temp = calc_color_deviation_uv(uv_coords)
        print(f"  色偏差 (Duv): {color_deviation:.6f}")
        print(f"  三角垂足插值CCT: {closest_temp:.1f} K")
    except Exception as e:
        print(f"  色偏差计算出错: {e}")
    
    # 检查约束满足情况
    print(f"\n约束满足情况:")
    cct_satisfied = 6000 <= results['CCT'] <= 7000
    rf_satisfied = results['Rf'] > 88
    rg_satisfied = 95 <= results['Rg'] <= 105
    
    print(f"  CCT ∈ [6000K, 7000K]: {'✓' if cct_satisfied else '✗'}")
    print(f"  Rf > 88: {'✓' if rf_satisfied else '✗'}")
    print(f"  Rg ∈ [95, 105]: {'✓' if rg_satisfied else '✗'}")
    
    # 绘制结果
    optimizer.plot_results(results)
    
    return results

if __name__ == "__main__":
    results = main()