import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob

# Tạo thư mục lưu mô hình nếu chưa có
os.makedirs("trained_models", exist_ok=True)

# Định nghĩa mô hình mạng nơ-ron đơn giản với 2 lớp Conv và 1 lớp Dense
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 4096)  # Mỗi ô trên bàn cờ 64x64 = 4096 nước đi
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        return self.softmax(x)

# Khởi tạo model, loss function và optimizer
model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Tìm tất cả các batch dữ liệu
batch_files = glob.glob("train_models/chess_training_data_batch_*.npz")
batch_files.sort()  # sắp xếp để huấn luyện theo thứ tự

# Huấn luyện mô hình qua nhiều epoch
epochs = 5
for epoch in range(epochs):
    print(f"\n Epoch {epoch + 1}/{epochs}")
    total_loss = 0
    for file in batch_files:
        print(f"   🔄 Đang huấn luyện với: {file}")
        data = np.load(file)
        X = torch.tensor(data["X"], dtype=torch.float32).permute(0, 3, 1, 2)  # (B, 12, 8, 8)
        y = torch.tensor(np.argmax(data["y"], axis=1), dtype=torch.long)     # (B,)

        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f" Tổng loss epoch {epoch + 1}: {total_loss:.4f}")

torch.save(model.state_dict(), "trained_models/chess_ai_model.pth")
print(" Đã lưu mô hình vào: trained_models/chess_ai_model.pth")
