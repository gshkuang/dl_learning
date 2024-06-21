import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms


# 定义教师网络
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 定义学生网络
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def distillation_loss(y, teacher_scores, label,T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T, dim=1),
                          F.softmax(teacher_scores/T, dim=1)) * (T*T * alpha) + F.cross_entropy(y, label) * (1. - alpha)

teacher_net = TeacherNet()
student_net = StudentNet()


# 定义数据预处理操作：转换为Tensor并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 下载并加载训练集
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载并加载测试集
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(student_net.parameters())
criterion = nn.CrossEntropyLoss()
for epoch in range(30):  # 举例，训练2个epoch
    train_loss=0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = student_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 使用训练好的教师模型测试学生模型
        correct = 0
        total = 0
        train_loss+=loss.item()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = student_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('train loss {:.2f} Accuracy on test set: {:.2f}%'.format(train_loss/len(trainloader),accuracy))
# 使用训练好的教师模型训练学生模型
student_net = StudentNet()
optimizer = torch.optim.Adam(student_net.parameters())

for i in range(30):
    train_loss=0
    for inputs, labels in trainloader:
        teacher_scores = teacher_net(inputs)
        student_scores = student_net(inputs)
        
        loss = distillation_loss(student_scores, teacher_scores,labels, T=2.0, alpha=0.5)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

    # 使用训练好的教师模型测试学生模型
    student_net.eval()  # Set the student model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = student_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('train loss {:.2f} Accuracy on test set: {:.2f}%'.format(train_loss/len(trainloader),accuracy))
