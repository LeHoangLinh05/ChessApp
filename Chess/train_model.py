import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
from tqdm import tqdm

os.makedirs("D:/KieuQuy/Documents/AI/Chess/Chess/trained_models", exist_ok=True)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 4096)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return self.softmax(x)

model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

batch_files = glob.glob(r"D:/KieuQuy/Documents/AI/Chess/Chess/train_models/chess_training_data_batch_*.npz")
batch_files.sort()
batch_files = batch_files[:100] 

print(f"🔍 Tìm thấy {len(batch_files)} batch.")

if len(batch_files) == 0:
    print("❌ Không tìm thấy dữ liệu để huấn luyện. Hãy kiểm tra đường dẫn hoặc dữ liệu.")
    exit()

epochs = 2
for epoch in range(epochs):
    print(f"\n🎯 Epoch {epoch + 1}/{epochs}")
    total_loss = 0
    for file in tqdm(batch_files, desc=f"Epoch {epoch + 1}"):
        data = np.load(file)
        X = torch.tensor(data["X"], dtype=torch.float32).permute(0, 3, 1, 2)
        y = torch.tensor(np.argmax(data["y"], axis=1), dtype=torch.long)

        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"✅ Tổng loss epoch {epoch + 1}: {total_loss:.4f}")

torch.save(model.state_dict(), "D:/KieuQuy/Documents/AI/Chess/Chess/trained_models/chess_ai_model.pth")
print("📦 Đã lưu mô hình tại: trained_models/chess_ai_model.pth")
