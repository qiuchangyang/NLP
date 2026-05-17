import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm # 用于显示极其酷炫的进度条
from sklearn.metrics import classification_report, f1_score

# 导入我们自己写的模块
from data_loader.dataset import ABSADataset
from models.main_model import ABSAMainModel

def train_and_evaluate():
    # ==========================================
    # 1. 全局超参数设置 (Hyperparameters)
    # ==========================================
    EPOCHS = 5                  # 训练轮数
    BATCH_SIZE = 8              # 每次喂给显卡的数据量 (如果显存爆了，改小到 4 或 2)
    LEARNING_RATE = 2e-5        # 学习率 (RoBERTa 微调的黄金学习率通常在 1e-5 到 5e-5 之间)
    MAX_LEN = 128               # 句子最大截断长度
    NUM_TAGS = 7                # BIO 标签总数
    TOKENIZER_NAME = 'roberta-base' 
    
    # 获取绝对路径，防止路径报错
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_JSON = os.path.join(BASE_DIR, "data", "processed", "laptops_train.json")
    # 假设你还有一个测试集 (如果没有，可以用 train.json 临时替代跑通流程)
    TEST_JSON = os.path.join(BASE_DIR, "data", "processed", "laptops_test.json") 
    
    # 自动检测设备：有 Nvidia 显卡就用 CUDA，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用的计算设备: {device}")

    # ==========================================
    # 2. 准备数据管道 (Data Pipeline)
    # ==========================================
    print("\n📦 正在构建训练集和测试集 (这可能需要加载 spaCy，请稍候)...")
    train_dataset = ABSADataset(TRAIN_JSON, TOKENIZER_NAME, MAX_LEN)
    # 此处假设你有测试集，如果没有，将 TEST_JSON 换成 TRAIN_JSON 体验流程
    test_dataset = ABSADataset(TRAIN_JSON, TOKENIZER_NAME, MAX_LEN) 
    
    # DataLoader 负责打乱数据并打包成 Batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. 引擎点火 (Model & Optimizer)
    # ==========================================
    print("\n🧠 正在加载主模型...")
    model = ABSAMainModel(
        model_name_or_path=TOKENIZER_NAME, 
        num_tags=NUM_TAGS, 
        gcn_out_dim=300
    ).to(device) # 把几百兆的模型塞进显卡显存里
    
    # 优化器：AdamW 是目前 NLP 领域的标配，负责根据梯度更新权重
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0.0 # 记录历史最佳 F1 分数

    # ==========================================
    # 4. 训练大循环 (The Training Loop)
    # ==========================================
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*20} Epoch {epoch}/{EPOCHS} {'='*20}")
        
        # ----------------------------------
        # 4.1 训练阶段 (Train Phase)
        # ----------------------------------
        model.train() # 极其重要：开启 Dropout 和 BatchNorm 的训练模式
        total_train_loss = 0
        
        # tqdm 包装 loader，生成动态进度条
        train_bar = tqdm(train_loader, desc=f"Training")
        
        for batch in train_bar:
            # 第一步：把数据从内存转移到显卡 (device) 上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            # 第二步：清空上一个 Batch 残留的梯度
            optimizer.zero_grad()
            
            # 第三步：前向传播 (Forward) -> 走完你的 RoBERTa+GCN+CRF 拿到 Loss
            loss = model(input_ids, attention_mask, adj_matrix, labels=labels)
            
            # 第四步：反向传播 (Backward) -> 算子导数，魔法发生的地方
            loss.backward()
            
            # 第五步：参数更新 -> 优化器让模型变聪明一点点
            optimizer.step()
            
            total_train_loss += loss.item()
            # 实时更新进度条上的 Loss 显示
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"🔥 Epoch {epoch} 训练完毕 | 平均 Loss: {avg_train_loss:.4f}")
        
        # ----------------------------------
        # 4.2 评估阶段 (Evaluation Phase)
        # ----------------------------------
        model.eval() # 极其重要：关闭 Dropout，保证预测的稳定性
        all_preds = []
        all_trues = []
        
        print(f"🔎 正在评估测试集...")
        with torch.no_grad(): # 极其重要：关闭梯度计算，省一半显存，提速一倍
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                adj_matrix = batch['adj_matrix'].to(device)
                labels = batch['labels'].to(device) # 测试集的真实标签
                
                # 预测模式，不传 labels，拿到维特比解码的最佳路径
                batch_preds = model(input_ids, attention_mask, adj_matrix, labels=None)
                
                # CRF 解码出来的是 List[List[int]]，长度去掉了 PAD
                # 我们要把真实的 labels 也去掉 PAD 提出来，拼成一维长数组算 F1
                for i in range(len(batch_preds)):
                    pred_path = batch_preds[i] # 这一句话的预测标签
                    true_path = labels[i][:len(pred_path)].cpu().numpy().tolist() # 截取真实标签对应的有效长度
                    
                    all_preds.extend(pred_path)
                    all_trues.extend(true_path)
        
        # ==========================================
        # 5. 算分与存盘 (Metrics & Checkpointing)
        # ==========================================
        # 注意：在真实算 F1 时，我们通常不关心 'O' 标签（背景词），所以可以过滤掉 0
        # 这里的 F1 用 macro，综合考虑各个正负向实体的识别准确率
        f1 = f1_score(all_trues, all_preds, average='macro')
        print(f"📊 当前 Epoch F1 Score: {f1:.4f}")
        
        # 如果这是历史最好成绩，存盘！
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(BASE_DIR, "best_model.pth")
            print(f"🎉 发现新高分！保存模型权重至 -> {save_path}")
            # 保存模型的 state_dict (纯权重字典，文件最小，最安全)
            torch.save(model.state_dict(), save_path)
            
    print("\n✅ 训练全部结束！")
    print(f"🏆 历史最佳 F1 分数: {best_f1:.4f}")

if __name__ == "__main__":
    train_and_evaluate()