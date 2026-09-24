import torch
import pickle
from model import ResNet50   # 用你项目里的 ResNet50 定义

# ====== 1. 读入外部人脸 ResNet50 权重（pkl + numpy） ======
# 把这里改成你下载的 pkl 文件路径
ext_path = r"D:\RA-UWML-AU-Pytorch-master\RA-UWML-AU-Pytorch-master+\models\resnet50_ft_weight.pkl"

print(f"Loading external weights from: {ext_path}")

# 这一步用的是 Python 的 pickle，而不是 torch.load
with open(ext_path, "rb") as f:
    obj = f.read()

# 原仓库的格式：一个 dict，值是 numpy.ndarray
np_state = pickle.loads(obj, encoding="latin1")
print("Loaded object type:", type(np_state))
print("Example keys:", list(np_state.keys())[:10])

# 把 numpy 参数转成 torch.Tensor，并去掉可能的 'module.' 前缀
clean_state = {}
for k, v in np_state.items():
    if k.startswith("module."):
        new_k = k[len("module."):]
    else:
        new_k = k
    clean_state[new_k] = torch.from_numpy(v)

print(f"External state_dict params: {len(clean_state)}")

# ====== 2. 构建你自己的 ResNet50 backbone ======
# strides 要和你训练脚本里 opts.strides 一样！！
# 比如 train_disfa.py 里 parser.add_argument('--strides', default='1,2,2,2')
strides = "1,2,2,2"   # ← 如果你训练时改了，这里也要跟着改

backbone = ResNet50(strides)
model_state = backbone.state_dict()
print(f"Your ResNet50 params: {len(model_state)}")

# ====== 3. 只拷贝“名字一样且 shape 一致”的参数 ======
loaded = 0
skipped = []

for k, v in clean_state.items():
    if k in model_state and model_state[k].shape == v.shape:
        model_state[k] = v
        loaded += 1
    else:
        skipped.append(k)

backbone.load_state_dict(model_state)
print(f"Matched and loaded {loaded} params.")
print(f"Skipped {len(skipped)} params (shape/name mismatch，正常现象，比如 fc 层、conv1 等)。")

# ====== 4. 存成 AUResnet 能用的 pretrain_backbone 文件 ======
save_path = "../models/pretrain_backbone_resnet50_face.pth"
torch.save({"net_state_dict": backbone.state_dict()}, save_path)
print(f"Saved converted backbone to: {save_path}")
