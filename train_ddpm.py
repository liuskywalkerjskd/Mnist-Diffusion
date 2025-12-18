import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
# 导入模型定义
from ddpm_model import ContextUnet, DDPM, ddpm_schedules

def main():
    # --- 超参数 ---
    n_epoch = 30
    batch_size = 256
    n_T = 400
    n_feat = 128
    n_classes = 10
    lrate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 结果保存路径
    save_dir = "./weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ddpm_mnist.pth")

    print(f"Using device: {device}")

    # --- 模型初始化 ---
    ddpm_constants = ddpm_schedules(1e-4, 0.02, n_T)
    
    model = ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes).to(device)
    ddpm = DDPM(nn_model=model, betas=ddpm_constants, n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lrate)

    # --- 数据加载 ---
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # --- 训练循环 ---
    print("Start Training Conditional DDPM...")
    for epoch in range(n_epoch):
        ddpm.train()
        optim_loss = 0
        
        for x, c in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            c = F.one_hot(c, num_classes=10).float().to(device)
            
            loss = ddpm(x, c)
            
            loss.backward()
            optimizer.step()
            optim_loss += loss.item()

        avg_loss = optim_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {avg_loss:.4f}")
        
        # 保存中间结果
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    # 保存最终模型
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Model saved to {save_path}")

if __name__ == "__main__":
    main()