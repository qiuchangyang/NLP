# evaluate.py
import torch
from tqdm import tqdm
from utils.metrics import compute_f1_score

def evaluate_model(model, dataloader, device):
    """
    在给定数据加载器上评估模型性能
    """
    model.eval() # 切换为评估模式
    all_preds = []
    all_trues = []
    
    with torch.no_grad(): # 关闭梯度计算，节省显存
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # 将数据推入显卡
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['labels'].to(device)
            
            # CRF 解码预测
            batch_preds = model(input_ids, attention_mask, adj_matrix, labels=None)
            
            # 剥离 PAD 并对齐长度
            for i in range(len(batch_preds)):
                pred_path = batch_preds[i] 
                true_path = labels[i][:len(pred_path)].cpu().numpy().tolist() 
                
                all_preds.extend(pred_path)
                all_trues.extend(true_path)
                
    # 调用 utils 中的计算工具
    f1 = compute_f1_score(all_trues, all_preds, average='macro')
    return f1