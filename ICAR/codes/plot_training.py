"""
训练日志可视化脚本
用于监控ICC和MAE变化趋势
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# 修改为你的结果目录
save_dir = r"D:\RA-UWML-AU-Pytorch-master\RA-UWML-AU-Pytorch-master+allimf\results\DISFA_IMF_CCNN\0001"

def plot_training_curves():
    train_csv = os.path.join(save_dir, 'train.csv')
    valid_csv = os.path.join(save_dir, 'valid.csv')

    if not os.path.exists(train_csv):
        print(f"训练日志不存在: {train_csv}")
        return

    # 读取训练日志
    try:
        train_df = pd.read_csv(train_csv, header=None, names=['iter', 'loss', 'icc', 'mae'])
        print(f"训练记录数: {len(train_df)}")
    except Exception as e:
        print(f"读取训练日志失败: {e}")
        return

    # 读取验证日志
    valid_df = None
    if os.path.exists(valid_csv):
        try:
            # 验证日志格式: iter_num, icc, mae, AU1_ICC, ..., AU12_ICC, AU1_MAE, ..., AU12_MAE
            valid_df = pd.read_csv(valid_csv, header=None)
            valid_df.columns = ['iter', 'icc', 'mae'] + \
                              [f'AU{i}_ICC' for i in [1,2,4,5,6,9,12,15,17,20,25,26]] + \
                              [f'AU{i}_MAE' for i in [1,2,4,5,6,9,12,15,17,20,25,26]]
            print(f"验证记录数: {len(valid_df)}")
        except Exception as e:
            print(f"读取验证日志失败: {e}")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss曲线
    axes[0, 0].plot(train_df['iter'], train_df['loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ICC曲线
    axes[0, 1].plot(train_df['iter'], train_df['icc'], label='Train ICC', alpha=0.7)
    if valid_df is not None and len(valid_df) > 0:
        axes[0, 1].plot(valid_df['iter'], valid_df['icc'], 'ro-', label='Valid ICC', markersize=8)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('ICC')
    axes[0, 1].set_title('ICC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MAE曲线
    axes[1, 0].plot(train_df['iter'], train_df['mae'], label='Train MAE', alpha=0.7)
    if valid_df is not None and len(valid_df) > 0:
        axes[1, 0].plot(valid_df['iter'], valid_df['mae'], 'ro-', label='Valid MAE', markersize=8)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Per-AU ICC (仅验证集)
    if valid_df is not None and len(valid_df) > 0:
        au_names = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
        latest_icc = [valid_df.iloc[-1][f'AU{i}_ICC'] for i in au_names]
        axes[1, 1].bar(range(len(au_names)), latest_icc)
        axes[1, 1].set_xticks(range(len(au_names)))
        axes[1, 1].set_xticklabels([f'AU{i}' for i in au_names], rotation=45)
        axes[1, 1].set_ylabel('ICC')
        axes[1, 1].set_title(f'Per-AU ICC (Latest Validation)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'No validation data yet',
                       ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"图表已保存到: {os.path.join(save_dir, 'training_curves.png')}")
    plt.show()

    # 打印当前最优结果
    print("\n=== 当前训练状态 ===")
    print(f"最新训练 - Loss: {train_df.iloc[-1]['loss']:.4f}, ICC: {train_df.iloc[-1]['icc']:.4f}, MAE: {train_df.iloc[-1]['mae']:.4f}")
    if valid_df is not None and len(valid_df) > 0:
        best_icc_idx = valid_df['icc'].idxmax()
        print(f"最佳验证 - ICC: {valid_df.iloc[best_icc_idx]['icc']:.4f} (Iter {int(valid_df.iloc[best_icc_idx]['iter'])})")
        print(f"         - MAE: {valid_df.iloc[best_icc_idx]['mae']:.4f}")

if __name__ == '__main__':
    plot_training_curves()
