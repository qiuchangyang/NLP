import json
import os
from skimage import data_dir
import torch
from torch.optim import AdamW
from tqdm import tqdm 
from models.main_model import ABSAMainModel
from evaluate import evaluate_model
from data_loader.dataset import ABSADataset
from torch.utils.data import DataLoader

def main():
    # 1. 超参数配置
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5 
    MAX_LEN = 128
    NUM_TAGS = 7 
    TOKENIZER_NAME = 'roberta-base' 
    SEED = 42 # 将 Seed 提升为全局配置
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_JSON = os.path.join(BASE_DIR, "data", "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [INIT] 设备: {device}")

    print("\n📦 正在加载静态数据集...")
    train_ds = ABSADataset(os.path.join(DATA_JSON,"train.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)
    val_ds = ABSADataset(os.path.join(DATA_JSON,"val.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)
    test_ds = ABSADataset(os.path.join(DATA_JSON,"test.json"), tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 实例化模型与优化器
    model = ABSAMainModel(model_name_or_path=TOKENIZER_NAME, num_tags=NUM_TAGS, gcn_out_dim=300).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_f1 = 0.0 

    # 4. 训练大循环
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*20} Epoch {epoch}/{EPOCHS} {'='*20}")
        
        # --- 4.1 核心训练逻辑 ---
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, adj_matrix, labels=labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    print(f"🔥 [TRAIN] 平均 Loss: {total_loss/len(train_loader):.4f}")
        
    # --- 4.2 调用独立的评估模块 ---
    # ⚠️ 注意：这里假设你的 evaluate_model 已经被修改为返回四个指标
    val_p, val_r, val_f1, val_acc = evaluate_model(model, val_loader, device)
    
    print(f"📊 [VALIDATION] Acc: {val_acc:.4f} | Precision: {val_p:.4f} | Recall: {val_r:.4f} | F1: {val_f1:.4f}")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        # 可以选择在破纪录时也顺便打印一下这一轮的其他分数
        print(f"🌟 新纪录！当前最佳 F1: {best_val_f1:.4f} (Acc: {val_acc:.4f})")
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"🎉 验证集创新高！权重已保存至: {BEST_MODEL_PATH}")
            
    # 5. 最终盲测
    print("\n" + "*"*40)
    print("🏆 开始测试集最终盲测...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_p, test_r, test_f1, test_acc = evaluate_model(model, test_loader, device)
    
    print(f"✅ 终极无偏测试结果:")
    print(f"   - 准确率 (Accuracy) : {test_acc:.4f}")
    print(f"   - 精确率 (Precision): {test_p:.4f}")
    print(f"   - 召回率 (Recall)   : {test_r:.4f}")
    print(f"   - 宏平均 F1 Score   : {test_f1:.4f}")

if __name__ == "__main__":
    main()