import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon, friedmanchisquare
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def fill_missing(sequence):
    """
    填补睡眠阶段编码序列的缺失值（NaN）。
    此函数借鉴自test.py，保证了缺失值处理的鲁棒性。
    """
    seq = sequence.tolist()
    n = len(seq)
    if n == 0: return []

    # 识别所有连续缺失段
    missing_segments = []
    current_start = None
    for i in range(n):
        if pd.isna(seq[i]):
            if current_start is None: current_start = i
            if i == n - 1: missing_segments.append((current_start, i))
        else:
            if current_start is not None:
                missing_segments.append((current_start, i - 1))
                current_start = None

    # 处理每个缺失段
    for start, end in missing_segments:
        length = end - start + 1
        fill_val = 4  # 默认填充为清醒

        if start == 0: # 首段缺失
            if end + 1 < n: fill_val = seq[end + 1]
        elif end == n - 1: # 末段缺失
            if start - 1 >= 0: fill_val = seq[start - 1]
        else: # 中间缺失
            prev_val, next_val = seq[start - 1], seq[end + 1]
            if length <= 2:
                fill_val = round((prev_val + next_val) / 2)
                seq[start] = fill_val
                if length == 2:
                    seq[end] = round((fill_val + next_val) / 2)
                continue
            else: # 长缺失段
                fill_val = prev_val # 简化规则，使用前一个值填充
        
        for i in range(start, end + 1):
            seq[i] = fill_val
            
    return [int(x) for x in seq]

def calculate_metrics(sequence):
    """
    根据处理后的睡眠阶段编码序列计算六个睡眠指标。
    """
    n = len(sequence)
    total_bed_time = n * 0.5  # 总卧床时间（分钟）

    non_awake_indices = [i for i, val in enumerate(sequence) if val != 4]
    
    if not non_awake_indices:
        TST, i_first, i_last = 0.0, None, None
    else:
        i_first, i_last = non_awake_indices[0], non_awake_indices[-1]
        TST = (i_last - i_first + 1) * 0.5

    SE = (TST / total_bed_time) * 100 if total_bed_time > 0 else 0.0
    SOL = i_first * 0.5 if i_first is not None else total_bed_time
    
    N3_count = sum(1 for val in sequence if val == 3)
    REM_count = sum(1 for val in sequence if val == 5)
    
    N3p = (N3_count * 0.5 / TST) * 100 if TST > 0 else 0.0
    REMp = (REM_count * 0.5 / TST) * 100 if TST > 0 else 0.0

    Awakenings = 0
    if i_first is not None and i_first < i_last:
        for i in range(i_first + 1, i_last + 1):
            if sequence[i] == 4 and sequence[i-1] != 4:
                Awakenings += 1

    return {
        '总睡眠时长(TST)': round(TST, 2), '睡眠效率(SE)': round(SE, 2),
        '入睡潜伏期(SOL)': round(SOL, 2), '深睡眠N3比例(N3p)': round(N3p, 2),
        'REM睡眠比例(REMp)': round(REMp, 2), '夜间醒来次数(Awakenings)': int(Awakenings)
    }

def process_all_data(df_wide):
    """
    处理所有数据，将其从宽格式转换为长格式的指标DataFrame。
    """
    results = []
    environments = ['助眠灯', '普通LED', '黑暗']
    
    for subject_id in range(1, 12):  # 11个受试者
        for env_idx, environment in enumerate(environments):
            col_idx = (subject_id - 1) * 3 + env_idx
            if col_idx < len(df_wide.columns):
                sequence = df_wide.iloc[:, col_idx]
                filled_sequence = fill_missing(sequence)
                metrics = calculate_metrics(filled_sequence)
                
                record = {'Subject_ID': subject_id, 'Environment': environment}
                record.update(metrics)
                results.append(record)
                
    return pd.DataFrame(results)

def run_statistical_analysis(df_long, metric):
    """
    对单个指标进行完整的统计检验。
    """
    data_a = df_long[df_long['Environment'] == '助眠灯'][metric]
    data_b = df_long[df_long['Environment'] == '普通LED'][metric]
    data_c = df_long[df_long['Environment'] == '黑暗'][metric]

    # 正态性检验
    stat, p_shapiro = shapiro(df_long[metric])
    is_normal = p_shapiro > 0.05

    results = {'metric': metric, 'is_normal': is_normal, 'p_shapiro': p_shapiro}

    # 主检验
    if is_normal:
        aov = AnovaRM(data=df_long, depvar=metric, subject='Subject_ID', within=['Environment']).fit()
        # Robustly find the p-value column to handle different statsmodels versions
        p_value_col = None
        possible_p_cols = ['p-value', 'PR(>F)', 'Pr(>F)', 'Pr > F']
        for col in possible_p_cols:
            if col in aov.anova_table.columns:
                p_value_col = col
                break
        
        if p_value_col is None:
             raise KeyError(f"Could not find p-value column in AnovaRM results. Available columns: {aov.anova_table.columns}")

        p_main = aov.anova_table[p_value_col][0]
        results.update({'test_type': 'RM-ANOVA', 'p_value': p_main})
        if p_main < 0.05:
            # 事后检验 (配对t检验 + Bonferroni校正)
            p_ab = ttest_rel(data_a, data_b).pvalue * 3
            p_ac = ttest_rel(data_a, data_c).pvalue * 3
            p_bc = ttest_rel(data_b, data_c).pvalue * 3
            results.update({'p_ab': p_ab, 'p_ac': p_ac, 'p_bc': p_bc})
    else:
        stat, p_main = friedmanchisquare(data_a, data_b, data_c)
        results.update({'test_type': 'Friedman', 'p_value': p_main})
        if p_main < 0.05:
            # 事后检验 (Wilcoxon + Bonferroni校正)
            p_ab = wilcoxon(data_a, data_b).pvalue * 3
            p_ac = wilcoxon(data_a, data_c).pvalue * 3
            p_bc = wilcoxon(data_b, data_c).pvalue * 3
            results.update({'p_ab': p_ab, 'p_ac': p_ac, 'p_bc': p_bc})
            
    return results

def generate_plots(df_long, metrics_to_plot, output_path='results/sleep_metrics_comparison.png'):
    """
    为所有指标生成箱形图并保存。
    """
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        sns.boxplot(x='Environment', y=metric, data=df_long, ax=axes[i],
                    order=['助眠灯', '普通LED', '黑暗'])
        sns.stripplot(x='Environment', y=metric, data=df_long, ax=axes[i],
                      color='black', jitter=0.2, size=4,
                      order=['助眠灯', '普通LED', '黑暗'])
        axes[i].set_title(f'{metric} 在不同光照下的分布', fontsize=14)
        axes[i].set_xlabel('光照环境', fontsize=12)
        axes[i].set_ylabel('值', fontsize=12)

    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3.0)
    plt.suptitle('不同光照环境对各项睡眠指标影响的可视化分析', fontsize=20, y=1.03)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"分析图表已保存至: {output_path}")

def main():
    """
    主函数：执行完整的分析流程。
    """
    # 1. 读取和处理数据
    try:
        df_wide = pd.read_excel("data.xlsx", sheet_name="Problem 4", header=0)
        # 删除第一行（时间标识）
        df_wide = df_wide.drop(0).reset_index(drop=True)
    except FileNotFoundError:
        print("错误: 未找到数据文件 'data.xlsx'。")
        return
    
    metrics_df = process_all_data(df_wide)
    
    # 2. 执行统计分析
    metrics_list = list(metrics_df.columns)[2:]
    stats_results = [run_statistical_analysis(metrics_df, metric) for metric in metrics_list]
    stats_df = pd.DataFrame(stats_results).round(4)

    # 3. 生成和保存结果
    # 创建results文件夹
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # 保存指标数据
    metrics_output_path = 'results/sleep_metrics_calculated.xlsx'
    metrics_df.to_excel(metrics_output_path, index=False)
    print(f"计算的睡眠指标已保存至: {metrics_output_path}")

    # 保存统计结果
    stats_output_path = 'results/statistical_analysis_summary.xlsx'
    stats_df.to_excel(stats_output_path, index=False)
    print(f"统计分析摘要已保存至: {stats_output_path}")

    # 4. 生成图表
    generate_plots(metrics_df, metrics_list)

    # 5. 打印总结
    print("\n" + "="*80)
    print("分析总结报告")
    print("="*80)
    print("详细的统计数据和图表请查看 'results' 文件夹下的文件。")
    
    for _, row in stats_df.iterrows():
        metric = row['metric']
        p_main = row['p_value']
        print(f"\n--- 指标: {metric} ---")
        if p_main < 0.05:
            print(f"✓ 结论: 不同光照环境对 {metric} 有显著影响 (p={p_main:.4f})。")
            if 'p_ab' in row and not pd.isna(row['p_ab']) and row['p_ab'] < 0.05/3:
                mean_a = metrics_df[metrics_df['Environment'] == '助眠灯'][metric].mean()
                mean_b = metrics_df[metrics_df['Environment'] == '普通LED'][metric].mean()
                direction = "优于" if (mean_a > mean_b and metric not in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) or \
                                     (mean_a < mean_b and metric in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) else "差于"
                print(f"  - 助眠灯 vs 普通LED: 存在显著差异 (p={row['p_ab']:.4f})，助眠灯表现 {direction} 普通LED。")
            if 'p_ac' in row and not pd.isna(row['p_ac']) and row['p_ac'] < 0.05/3:
                mean_a = metrics_df[metrics_df['Environment'] == '助眠灯'][metric].mean()
                mean_c = metrics_df[metrics_df['Environment'] == '黑暗'][metric].mean()
                direction = "优于" if (mean_a > mean_c and metric not in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) or \
                                     (mean_a < mean_c and metric in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) else "差于"
                print(f"  - 助眠灯 vs 黑暗: 存在显著差异 (p={row['p_ac']:.4f})，助眠灯表现 {direction} 黑暗环境。")
        else:
            print(f"✗ 结论: 不同光照环境对 {metric} 无显著影响 (p={p_main:.4f})。")

if __name__ == "__main__":
    main()
