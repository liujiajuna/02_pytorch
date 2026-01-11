import torch

# 创建一个 2D 张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 张量的属性
print("Tensor:\n", tensor)
print("Shape:", tensor.shape)  # 获取形状
print("Size:", tensor.size())  # 获取形状（另一种方法）
print("Data Type:", tensor.dtype)  # 数据类型
print("Device:", tensor.device)  # 设备
print("Dimensions:", tensor.dim())  # 维度数
print("Total Elements:", tensor.numel())  # 元素总数
print("Requires Grad:", tensor.requires_grad)  # 是否启用梯度
print("Is CUDA:", tensor.is_cuda)  # 是否在 GPU 上
print("Is Contiguous:", tensor.is_contiguous())  # 是否连续存储

# 获取单元素值
single_value = torch.tensor(42)
print("Single Element Value:", single_value.item())

# 转置张量
tensor_T = tensor.T
print("Transposed Tensor:\n", tensor_T)

print("\n=== PyTorch 数据集与 DataLoader 示例 ===")

# 示例 1：使用 TensorDataset + DataLoader
features = torch.randn(10, 4)  # 10 个样本，每个样本 4 维特征
labels = torch.randint(0, 2, (10,))  # 二分类标签
tensor_dataset = torch.utils.data.TensorDataset(features, labels)
tensor_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=3, shuffle=True)

print("\n示例 1：TensorDataset")
for batch_idx, (batch_x, batch_y) in enumerate(tensor_loader):
    print(f"Batch {batch_idx} - X shape: {batch_x.shape}, y: {batch_y}")


# 示例 2：自定义 Map-style Dataset
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int = 8):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor([idx, idx + 1], dtype=torch.float32)
        y = torch.tensor(idx % 3, dtype=torch.long)
        return x, y


toy_dataset = ToyDataset(num_samples=8)
toy_loader = torch.utils.data.DataLoader(toy_dataset, batch_size=4, shuffle=False)

print("\n示例 2：自定义 Map-style Dataset")
for batch_idx, (batch_x, batch_y) in enumerate(toy_loader):
    print(f"Batch {batch_idx} - X: {batch_x}, y: {batch_y}")


# 示例 3：自定义 IterableDataset
class RangeIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(self.start, self.end):
            yield torch.tensor([i, i * 2], dtype=torch.float32), torch.tensor(i % 2)


iter_dataset = RangeIterableDataset(start=0, end=6)
iter_loader = torch.utils.data.DataLoader(iter_dataset, batch_size=2)

print("\n示例 3：自定义 IterableDataset")
for batch_idx, (batch_x, batch_y) in enumerate(iter_loader):
    print(f"Batch {batch_idx} - X: {batch_x}, y: {batch_y}")


# 示例 4：使用 collate_fn 处理可变长度序列
def pad_collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded, lengths, labels


variable_dataset = [
    (torch.tensor([1, 2, 3], dtype=torch.float32), torch.tensor(0)),
    (torch.tensor([4, 5], dtype=torch.float32), torch.tensor(1)),
    (torch.tensor([6], dtype=torch.float32), torch.tensor(0)),
]

variable_loader = torch.utils.data.DataLoader(variable_dataset, batch_size=2, collate_fn=pad_collate_fn)

print("\n示例 4：collate_fn 处理可变长度序列")
for batch_idx, (padded, lengths, batch_y) in enumerate(variable_loader):
    print(f"Batch {batch_idx} - Padded: {padded}, Lengths: {lengths}, y: {batch_y}")
