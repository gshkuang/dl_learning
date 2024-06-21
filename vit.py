import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H, W)
        return x.flatten(2).transpose(1, 2)  # (B, N, E)

class ViT(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_chans, embed_dim)

        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# 创建模型
model = ViT(patch_size=16, in_chans=1, embed_dim=768, num_heads=4, num_layers=6, num_classes=10)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为3通道
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))  # 对每个通道进行归一化
])

# 加载MNIST数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# # 加载预训练的ViT模型
# model = timm.create_model('vit_base_patch16_224', pretrained=True)

# # 修改模型的最后一层以适应MNIST数据集（10个类别）
# model.head = torch.nn.Linear(model.head.in_features, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())

# 训练模型
model.train()
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 测试模型
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 将批次的损失相加
        pred = output.argmax(dim=1, keepdim=True)  # 获取概率最高的预测
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))