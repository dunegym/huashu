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
    为所有指标生成多种类型的图表并保存。
    """
    n_metrics = len(metrics_to_plot)
    
    # 创建更大的图形来容纳多种图表类型
    fig = plt.figure(figsize=(24, 16))
    
    # 为每个指标创建3种图表：箱形图、小提琴图、条形图
    for i, metric in enumerate(metrics_to_plot):
        # 箱形图 + 散点图
        ax1 = plt.subplot(3, n_metrics, i + 1)
        sns.boxplot(x='Environment', y=metric, data=df_long, ax=ax1,
                    order=['助眠灯', '普通LED', '黑暗'], palette='Set2')
        sns.stripplot(x='Environment', y=metric, data=df_long, ax=ax1,
                      color='black', jitter=0.2, size=4,
                      order=['助眠灯', '普通LED', '黑暗'])
        ax1.set_title(f'{metric} - 箱形图', fontsize=12, fontweight='bold')
        ax1.set_xlabel('光照环境', fontsize=10)
        ax1.set_ylabel('值', fontsize=10)
        ax1.grid(alpha=0.3)
        
        # 小提琴图
        ax2 = plt.subplot(3, n_metrics, i + 1 + n_metrics)
        sns.violinplot(x='Environment', y=metric, data=df_long, ax=ax2,
                       order=['助眠灯', '普通LED', '黑暗'], palette='Set3',
                       inner='quartile')
        # 在小提琴图上叠加散点
        sns.stripplot(x='Environment', y=metric, data=df_long, ax=ax2,
                      color='white', jitter=0.15, size=3, alpha=0.8,
                      order=['助眠灯', '普通LED', '黑暗'])
        ax2.set_title(f'{metric} - 小提琴图', fontsize=12, fontweight='bold')
        ax2.set_xlabel('光照环境', fontsize=10)
        ax2.set_ylabel('值', fontsize=10)
        ax2.grid(alpha=0.3)
        
        # 均值条形图 + 误差棒
        ax3 = plt.subplot(3, n_metrics, i + 1 + 2*n_metrics)
        
        # 计算均值和标准误差
        summary_stats = df_long.groupby('Environment')[metric].agg(['mean', 'std', 'count']).reset_index()
        summary_stats['se'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
        summary_stats = summary_stats.set_index('Environment').reindex(['助眠灯', '普通LED', '黑暗'])
        
        bars = ax3.bar(range(len(summary_stats)), summary_stats['mean'], 
                       yerr=summary_stats['se'], capsize=5, 
                       color=['lightcoral', 'lightblue', 'lightgreen'],
                       alpha=0.7, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for j, (bar, mean_val, se_val) in enumerate(zip(bars, summary_stats['mean'], summary_stats['se'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + se_val,
                     f'{mean_val:.1f}±{se_val:.1f}',
                     ha='center', va='bottom', fontsize=9)
        
        ax3.set_title(f'{metric} - 均值±标准误', fontsize=12, fontweight='bold')
        ax3.set_xlabel('光照环境', fontsize=10)
        ax3.set_ylabel('均值', fontsize=10)
        ax3.set_xticks(range(3))
        ax3.set_xticklabels(['助眠灯', '普通LED', '黑暗'])
        ax3.grid(alpha=0.3, axis='y')

    plt.tight_layout(pad=2.0)
    plt.suptitle('不同光照环境对各项睡眠指标影响的综合可视化分析', fontsize=20, y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合分析图表已保存至: {output_path}")

def generate_separate_plots(df_long, metrics_to_plot):
    """
    为所有指标生成分离的图表类型并保存为独立文件。
    """
    n_metrics = len(metrics_to_plot)
    
    # 1. 箱形图
    fig = plt.figure(figsize=(20, 8))
    for i, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(2, 3, i + 1)
        sns.boxplot(x='Environment', y=metric, data=df_long, ax=ax,
                    order=['助眠灯', '普通LED', '黑暗'], palette='Set2')
        sns.stripplot(x='Environment', y=metric, data=df_long, ax=ax,
                      color='black', jitter=0.2, size=4,
                      order=['助眠灯', '普通LED', '黑暗'])
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('光照环境', fontsize=10)
        ax.set_ylabel('值', fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('睡眠指标箱形图分析', fontsize=16, y=0.98)
    plt.savefig('results/boxplots_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("箱形图已保存至: results/boxplots_comparison.png")
    
    # 2. 小提琴图
    fig = plt.figure(figsize=(20, 8))
    for i, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(2, 3, i + 1)
        sns.violinplot(x='Environment', y=metric, data=df_long, ax=ax,
                       order=['助眠灯', '普通LED', '黑暗'], palette='Set3',
                       inner='quartile')
        sns.stripplot(x='Environment', y=metric, data=df_long, ax=ax,
                      color='white', jitter=0.15, size=3, alpha=0.8,
                      order=['助眠灯', '普通LED', '黑暗'])
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('光照环境', fontsize=10)
        ax.set_ylabel('值', fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('睡眠指标小提琴图分析', fontsize=16, y=0.98)
    plt.savefig('results/violinplots_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("小提琴图已保存至: results/violinplots_comparison.png")
    
    # 3. 柱状图（均值条形图）
    fig = plt.figure(figsize=(20, 8))
    for i, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(2, 3, i + 1)
        
        # 计算均值和标准误差
        summary_stats = df_long.groupby('Environment')[metric].agg(['mean', 'std', 'count']).reset_index()
        summary_stats['se'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
        summary_stats = summary_stats.set_index('Environment').reindex(['助眠灯', '普通LED', '黑暗'])
        
        bars = ax.bar(range(len(summary_stats)), summary_stats['mean'], 
                      yerr=summary_stats['se'], capsize=5, 
                      color=['lightcoral', 'lightblue', 'lightgreen'],
                      alpha=0.7, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for j, (bar, mean_val, se_val) in enumerate(zip(bars, summary_stats['mean'], summary_stats['se'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + se_val,
                    f'{mean_val:.1f}±{se_val:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('光照环境', fontsize=10)
        ax.set_ylabel('均值', fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['助眠灯', '普通LED', '黑暗'])
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.suptitle('睡眠指标均值条形图分析', fontsize=16, y=0.98)
    plt.savefig('results/barplots_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("柱状图已保存至: results/barplots_comparison.png")

def generate_additional_plots(df_long, metrics_to_plot):
    """
    生成额外的分析图表
    """
    # 1. 相关性热力图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 为每个环境生成相关性矩阵
    environments = ['助眠灯', '普通LED', '黑暗']
    for i, env in enumerate(environments[:3]):
        if i < 3:
            row, col = i // 2, i % 2
            env_data = df_long[df_long['Environment'] == env][metrics_to_plot]
            corr_matrix = env_data.corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[row, col], fmt='.2f', square=True,
                       cbar_kws={'label': '相关系数'})
            axes[row, col].set_title(f'{env} - 指标相关性', fontsize=14, fontweight='bold')
    
    # 整体相关性矩阵
    overall_corr = df_long[metrics_to_plot].corr()
    sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0,
               ax=axes[1, 1], fmt='.2f', square=True,
               cbar_kws={'label': '相关系数'})
    axes[1, 1].set_title('整体 - 指标相关性', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("相关性热力图已保存至: results/correlation_heatmaps.png")
    
    # 2. 雷达图比较
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 计算每个环境的标准化均值
    env_means = {}
    for env in environments:
        env_data = df_long[df_long['Environment'] == env][metrics_to_plot]
        env_means[env] = env_data.mean()
    
    # 标准化数据（0-1范围）
    all_values = np.concatenate([env_means[env].values for env in environments])
    min_val, max_val = all_values.min(), all_values.max()
    
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    colors = ['red', 'blue', 'green']
    for i, env in enumerate(environments):
        values = [(val - min_val) / (max_val - min_val) for val in env_means[env].values]
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=env, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.split('(')[0] for metric in metrics_to_plot])
    ax.set_ylim(0, 1)
    ax.set_title('各环境睡眠指标雷达图比较', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.savefig('results/radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("雷达图比较已保存至: results/radar_comparison.png")
    
    # 3. 分布密度图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        for env in environments:
            env_data = df_long[df_long['Environment'] == env][metric]
            sns.kdeplot(data=env_data, ax=axes[i], label=env, alpha=0.7, linewidth=2)
        
        axes[i].set_title(f'{metric} - 概率密度分布', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('值', fontsize=10)
        axes[i].set_ylabel('密度', fontsize=10)
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/density_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("密度分布图已保存至: results/density_distributions.png")

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

    # 4. 生成分离的图表
    generate_separate_plots(metrics_df, metrics_list)
    
    # 5. 生成额外的分析图表
    generate_additional_plots(metrics_df, metrics_list)

    # 6. 打印总结
    print("\n" + "="*80)
    print("分析总结报告")
    print("="*80)
    print("详细的统计数据和图表请查看 'results' 文件夹下的文件。")
    print("生成的图表包括：")
    print("- 箱形图分析（boxplots_comparison.png）")
    print("- 小提琴图分析（violinplots_comparison.png）")
    print("- 柱状图分析（barplots_comparison.png）")
    print("- 指标相关性热力图（correlation_heatmaps.png）")
    print("- 各环境睡眠指标雷达图比较（radar_comparison.png）")
    print("- 概率密度分布图（density_distributions.png）")
    
    for _, row in stats_df.iterrows():
        metric = row['metric']
        p_main = row['p_value']
        test_type = row.get('test_type', 'Unknown')
        
        print(f"\n--- 指标: {metric} ---")
        print(f"检验方法: {test_type}")
        print(f"主检验p值: {p_main:.4f}")
        
        # 显示描述性统计
        data_助眠灯 = metrics_df[metrics_df['Environment'] == '助眠灯'][metric]
        data_普通LED = metrics_df[metrics_df['Environment'] == '普通LED'][metric]
        data_黑暗 = metrics_df[metrics_df['Environment'] == '黑暗'][metric]
        
        print(f"描述性统计 (均值 ± 标准差):")
        print(f"  助眠灯: {data_助眠灯.mean():.2f} ± {data_助眠灯.std():.2f}")
        print(f"  普通LED: {data_普通LED.mean():.2f} ± {data_普通LED.std():.2f}")
        print(f"  黑暗: {data_黑暗.mean():.2f} ± {data_黑暗.std():.2f}")
        
        if p_main < 0.05:
            print(f"✓ 结论: 不同光照环境对 {metric} 有显著影响 (p={p_main:.4f})。")
            
            # 事后检验结果
            if 'p_ab' in row and not pd.isna(row['p_ab']):
                print(f"事后检验结果 (Bonferroni校正):")
                p_ab = row['p_ab']
                p_ac = row.get('p_ac', np.nan)
                p_bc = row.get('p_bc', np.nan)
                
                print(f"  助眠灯 vs 普通LED: p={p_ab:.4f}", end="")
                if p_ab < 0.05:
                    mean_a = data_助眠灯.mean()
                    mean_b = data_普通LED.mean()
                    direction = "优于" if (mean_a > mean_b and metric not in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) or \
                                         (mean_a < mean_b and metric in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) else "差于"
                    print(f" [显著差异，助眠灯{direction}普通LED]")
                else:
                    print(f" [无显著差异]")
                
                if not pd.isna(p_ac):
                    print(f"  助眠灯 vs 黑暗: p={p_ac:.4f}", end="")
                    if p_ac < 0.05:
                        mean_a = data_助眠灯.mean()
                        mean_c = data_黑暗.mean()
                        direction = "优于" if (mean_a > mean_c and metric not in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) or \
                                             (mean_a < mean_c and metric in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) else "差于"
                        print(f" [显著差异，助眠灯{direction}黑暗环境]")
                    else:
                        print(f" [无显著差异]")
                
                if not pd.isna(p_bc):
                    print(f"  普通LED vs 黑暗: p={p_bc:.4f}", end="")
                    if p_bc < 0.05:
                        mean_b = data_普通LED.mean()
                        mean_c = data_黑暗.mean()
                        direction = "优于" if (mean_b > mean_c and metric not in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) or \
                                             (mean_b < mean_c and metric in ['入睡潜伏期(SOL)', '夜间醒来次数(Awakenings)']) else "差于"
                        print(f" [显著差异，普通LED{direction}黑暗环境]")
                    else:
                        print(f" [无显著差异]")
        else:
            print(f"✗ 结论: 不同光照环境对 {metric} 无显著影响 (p={p_main:.4f})。")
    
    print("\n" + "="*80)
    print("统计说明:")
    print("- 正态性检验: Shapiro-Wilk检验 (p>0.05为正态)")
    print("- 正态数据: 重复测量方差分析(RM-ANOVA) + 配对t检验")
    print("- 非正态数据: Friedman检验 + Wilcoxon符号秩检验")
    print("- 多重比较校正: Bonferroni方法 (p值×3)")
    print("- 显著性水平: α=0.05")
    print("="*80)

if __name__ == "__main__":
    main()