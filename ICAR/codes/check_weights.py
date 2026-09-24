import torch

# 检查预训练权重
checkpoint_path = r"D:\RA-UWML-AU-Pytorch-master\RA-UWML-AU-Pytorch-master+allimf\models\checkpoint_100.pth"

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 查看checkpoint结构
print("\n=== Checkpoint Keys ===")
print(checkpoint.keys())

# 查看state_dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("\n=== Model State Dict (first 20 keys) ===")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        print(f"{i+1}. {key}: {state_dict[key].shape}")

    # 统计参数数量
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n=== Total Parameters: {total_params:,} ===")

    # 检查是否有 imf. 前缀
    imf_keys = [k for k in state_dict.keys() if k.startswith('imf.')]
    module_keys = [k for k in state_dict.keys() if k.startswith('module.')]

    print(f"\n=== Keys with 'imf.' prefix: {len(imf_keys)} ===")
    print(f"=== Keys with 'module.' prefix: {len(module_keys)} ===")

    if len(imf_keys) > 0:
        print("\nExample IMF keys:")
        for k in imf_keys[:5]:
            print(f"  - {k}")

    if len(module_keys) > 0:
        print("\nExample module keys:")
        for k in module_keys[:5]:
            print(f"  - {k}")

elif 'state_dict' in checkpoint:
    print("\nUsing 'state_dict' key")
    state_dict = checkpoint['state_dict']
    print(f"Total keys: {len(state_dict)}")
else:
    print("\nCheckpoint IS the state_dict")
    state_dict = checkpoint
    print(f"Total keys: {len(state_dict)}")

# 检查其他信息
if 'epoch' in checkpoint:
    print(f"\n=== Checkpoint Epoch: {checkpoint['epoch']} ===")
if 'loss' in checkpoint:
    print(f"=== Checkpoint Loss: {checkpoint['loss']} ===")
