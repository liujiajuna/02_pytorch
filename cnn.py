import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -----------------------------
# Global config
# -----------------------------
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

MODEL_PATH = "model.pth"
FORCE_RETRAIN = False  # True: always retrain and overwrite model.pth

CANVAS_SIZE = 280
BRUSH_THICKNESS = 20
WINDOW_NAME = "Draw (p:predict c:clear q:quit)"


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(data_root: str, batch_size: int, device: torch.device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

    pin = (device.type == "cuda")
    # num_workers 可按你的机器调整；Windows 下太大有时反而慢或有问题
    num_workers = 2

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )
    return train_loader, test_loader


# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 14x14 -> 14x14
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28 -> 14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14 -> 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        pred = torch.argmax(logits, dim=1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

    acc = 100.0 * correct / max(total, 1)
    return acc


def save_model(model, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model_weights(model, path: str, device: torch.device) -> bool:
    if not os.path.exists(path):
        return False

    try:
        # PyTorch 新版本可用 weights_only=True（更安全）
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # 老版本 PyTorch 不支持 weights_only
        state = torch.load(path, map_location=device)

    model.load_state_dict(state)
    return True


# -----------------------------
# Visualization
# -----------------------------
@torch.no_grad()
def visualize_predictions(model, test_loader, device, n_show: int = 6):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)
    logits = model(images)
    preds = torch.argmax(logits, dim=1).cpu()

    fig, axes = plt.subplots(1, n_show, figsize=(12, 4))
    for i in range(n_show):
        # 反归一化到 [0,1] 便于可视化
        img = images[i, 0].cpu() * MNIST_STD + MNIST_MEAN
        img = img.clamp(0, 1).numpy()

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {labels[i].item()}\nPred: {preds[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Interactive drawing preprocess (MNIST-like)
# -----------------------------
def _center_by_mass(img28: np.ndarray) -> np.ndarray:
    """
    img28: float32 [0,1], shape (28,28), background 0, foreground >0
    shift to make center-of-mass close to (14,14)
    """
    m = cv2.moments(img28)
    if abs(m["m00"]) < 1e-6:
        return img28

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    shift_x = 14.0 - cx
    shift_y = 14.0 - cy

    M = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    shifted = cv2.warpAffine(img28, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return shifted


def preprocess_canvas_mnist_like(canvas: np.ndarray, device: torch.device) -> torch.Tensor | None:
    """
    canvas: uint8 (CANVAS_SIZE, CANVAS_SIZE), background=0, stroke=255
    returns: torch tensor (1,1,28,28) normalized with MNIST mean/std
    """
    # 找非零区域
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # 扩边，避免数字贴边被截断
    pad = 20
    x0 = max(0, x0 - pad)
    x1 = min(canvas.shape[1] - 1, x1 + pad)
    y0 = max(0, y0 - pad)
    y1 = min(canvas.shape[0] - 1, y1 + pad)

    crop = canvas[y0:y1 + 1, x0:x1 + 1]

    # 等比缩放：最大边缩到 20 像素（MNIST 近似做法）
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad 到 28x28 并居中
    out = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    out[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # 轻微平滑（可选但常有帮助）
    out = cv2.GaussianBlur(out, (3, 3), 0)

    # 转 float [0,1]
    out_f = out.astype(np.float32) / 255.0

    # 按质心做细调平移，使更像 MNIST 的中心化
    out_f = _center_by_mass(out_f)

    # 归一化与训练一致
    tensor = torch.from_numpy(out_f).unsqueeze(0).unsqueeze(0)  # 1x1x28x28
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.to(device)


# -----------------------------
# Interactive UI
# -----------------------------
def interactive_predict_loop(model, device: torch.device):
    drawing = False
    last_pt = None
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_pt, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing and last_pt is not None:
            cv2.line(canvas, last_pt, (x, y), 255, thickness=BRUSH_THICKNESS)
            last_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if last_pt is not None:
                cv2.line(canvas, last_pt, (x, y), 255, thickness=BRUSH_THICKNESS)
            last_pt = None

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, draw)

    print("打开绘图窗口：左键绘制，按 p 预测，按 c 清空，按 q 退出")

    last_pred_text = ""

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        disp = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp, "p:predict  c:clear  q:quit", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if last_pred_text:
            cv2.putText(disp, last_pred_text, (8, CANVAS_SIZE - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):
            tensor = preprocess_canvas_mnist_like(canvas, device)
            if tensor is None:
                print("Canvas is empty. Draw something first.")
                last_pred_text = "Empty"
                continue

            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                prob = F.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(prob).item())
                topk = torch.topk(prob, k=3)

                top3 = [(int(topk.indices[i].item()), float(topk.values[i].item())) for i in range(3)]
                print(f"Prediction: {pred}, Top3: {top3}")

                last_pred_text = f"Pred: {pred}  ({top3[0][1]*100:.1f}%)"

        elif key == ord("c"):
            canvas[:] = 0
            last_pred_text = ""
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    # 用脚本所在目录，避免依赖 cwd
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "data")

    train_loader, test_loader = build_loaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
        device=device
    )

    model = SimpleCNN().to(device)

    need_train = True
    if os.path.exists(MODEL_PATH) and not FORCE_RETRAIN:
        try:
            ok = load_model_weights(model, MODEL_PATH, device)
            if ok:
                print(f"Loaded model from {MODEL_PATH}")
                need_train = False
        except Exception as e:
            print("Warning: failed to load existing model:", e)
            need_train = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if need_train:
        print("Training...")
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Loss: {train_loss:.4f}  Test Acc: {test_acc:.2f}%")

        try:
            save_model(model, MODEL_PATH)
            print(f"Saved trained model to {MODEL_PATH}")
        except Exception as e:
            print("Warning: failed to save model:", e)
    else:
        print("Skipping training since model is loaded.")

    # Final test accuracy
    final_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

    # Visualization
    visualize_predictions(model, test_loader, device, n_show=6)

    # Interactive
    interactive_predict_loop(model, device)


if __name__ == "__main__":
    main()
