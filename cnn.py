import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# 设备选择（CPU 或 CUDA）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 的均值/方差
])

# 加载 MNIST 数据集
data_root = os.path.join(os.getcwd(), 'data')
train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入32通道，输出64通道
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 展平后输入到全连接层
        self.fc2 = nn.Linear(128, 10)  # 10 个类别

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = x.view(-1, 64 * 7 * 7) # 展平
        x = F.relu(self.fc1(x))    # 全连接层 + ReLU
        x = self.fc2(x)            # 最后一层输出
        return x

# 创建模型实例及主流程封装，避免导入时自动执行
def main():
    # 创建模型实例
    model = SimpleCNN().to(device)
    model_path = 'model.pth'
    FORCE_RETRAIN = True  # 若为 True 则强制重新训练（覆盖 model.pth）
    if os.path.exists(model_path) and not FORCE_RETRAIN:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model from {model_path}')
        need_train = False
    else:
        need_train = True

    # 3. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 改用 Adam

    # 4. 模型训练
    num_epochs = 15  # 增加训练轮数
    if need_train:
        model.train()  # 设置模型为训练模式

        for epoch in range(num_epochs):
            total_loss = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失

                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # 保存训练好的模型参数
        try:
            torch.save(model.state_dict(), model_path)
            print(f'Saved trained model to {model_path}')
        except Exception as e:
            print('Warning: failed to save model:', e)
    else:
        print('Skipping training since model.pth loaded.')

    # 5. 模型测试
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 6. 可视化测试结果
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, 6, figsize=(12, 4))
    for i in range(6):
        # 图像用于显示时需回到 CPU 并转换为 numpy
        img_disp = images[i][0].cpu().numpy()
        axes[i].imshow(img_disp, cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        axes[i].axis('off')
    plt.show()

    # 7. 交互式手写输入与预测（使用鼠标绘制）
    drawing = False
    last_pt = None
    canvas = np.zeros((280, 280), dtype=np.uint8)

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_pt, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing and last_pt is not None:
            cv2.line(canvas, last_pt, (x, y), 255, thickness=20)
            last_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and last_pt is not None:
            drawing = False
            cv2.line(canvas, last_pt, (x, y), 255, thickness=20)

    def preprocess_canvas(img):
        # img: numpy uint8 28x28 or larger, foreground=255
        # Resize to 28x28, scale to [0,1], normalize to training normalization
        img28 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img28 = img28.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0)  # 1x1x28x28
        tensor = (tensor - 0.5) / 0.5
        return tensor.to(device)

    def interactive_predict_loop(model):
        cv2.namedWindow('Draw (p:predict c:clear q:quit)')
        cv2.setMouseCallback('Draw (p:predict c:clear q:quit)', draw)

        print('打开绘图窗口：左键绘制，按 p 预测，按 c 清空，按 q 退出')
        while True:
            disp = cv2.cvtColor(cv2.resize(canvas, (280, 280)), cv2.COLOR_GRAY2BGR)
            cv2.putText(disp, 'p:predict  c:clear  q:quit', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow('Draw (p:predict c:clear q:quit)', disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                img = cv2.resize(canvas, (28, 28))
                tensor = preprocess_canvas(img)
                model.eval()
                with torch.no_grad():
                    outputs = model(tensor)
                    _, pred = torch.max(outputs, 1)
                    print('Prediction:', pred.item())
                    # 临时在窗口显示预测结果
                    show = disp.copy()
                    cv2.putText(show, f'Pred: {pred.item()}', (5, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow('Draw (p:predict c:clear q:quit)', show)
                    cv2.waitKey(500)
            elif key == ord('c'):
                canvas[:] = 0
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()

    # 启动交互式预测循环（训练与测试后）
    interactive_predict_loop(model)


if __name__ == '__main__':
    main()