import torch
import torch.nn as nn
from transformers import AutoModel

# 导入我们之前一步步写好并测试通过的零件
from layers.gcn_layer import GCNLayer
from layers.attention import SelfAttentionLayer
from layers.crf_layer import CRFLayer

class ABSAMainModel(nn.Module):
    def __init__(self, model_name_or_path, num_tags=7, gcn_out_dim=300, dropout_rate=0.1):
        """
        ABSA 端到端主模型初始化 (RoBERTa + Self-Attention + GCN + CRF)
        
        :param model_name_or_path: 预训练语言模型的名称或本地路径 (如 'roberta-base')
        :param num_tags: 序列标注的标签总数 (BIO体系下为 7)
        :param gcn_out_dim: GCN 聚合后的特征维度 (映射到较低维度以防过拟合)
        :param dropout_rate: 随机失活率
        """
        super(ABSAMainModel, self).__init__()
        
        # 1. 加载真正的预训练语言模型大脑 (几百MB的参数权重在这里被真正加载)
        print(f"正在加载预训练语言模型权重: {model_name_or_path} ...")
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        
        # 获取预训练模型输出的特征维度 (例如 roberta-base 是 768)
        self.hidden_size = self.roberta.config.hidden_size
        
        # 2. 实例化自注意力层 (建立长距离语义相关性，对抗可能错误的句法树)
        self.attention = SelfAttentionLayer(
            hidden_size=self.hidden_size, 
            num_heads=8, 
            dropout_rate=dropout_rate
        )
        
        # 3. 实例化图卷积层 (将句法邻接矩阵的结构信息强行注入词向量)
        self.gcn = GCNLayer(
            in_features=self.hidden_size, 
            out_features=gcn_out_dim, 
            dropout_rate=dropout_rate
        )
        
        # 4. 分类器 / 发射线性层 (Classifier / Emission Layer)
        # 将 GCN 出来的特征维度 (gcn_out_dim) 映射到标签空间 (num_tags)
        # 它的输出就是 CRF 所需要的 “发射矩阵 (Emissions)”
        self.classifier = nn.Linear(gcn_out_dim, num_tags)
        
        # 5. 实例化条件随机场层 (掌控全局标签转移规则的守门员)
        self.crf = CRFLayer(num_tags=num_tags)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, adj_matrix, labels=None):
        """
        核心前向传播逻辑 (掌控数据流向)
        
        :param input_ids: 词ID序列，形状: [batch_size, seq_len]
        :param attention_mask: 语言模型注意力掩码，形状: [batch_size, seq_len]
        :param adj_matrix: 句法依赖邻接矩阵，形状: [batch_size, seq_len, seq_len]
        :param labels: 真实的BIO标签 (仅在训练时传入)，形状: [batch_size, seq_len]
        
        :return: 
            训练模式下 (labels不为None): 返回 CRF 算出的 Loss 标量
            预测模式下 (labels为None): 返回维特比解码出的最优标签路径列表 List[List[int]]
        """
        
        # ---------------------------------------------------------
        # 阶段一：通用语义提取 (RoBERTa)
        # ---------------------------------------------------------
        # input_ids & attention_mask: [batch_size, seq_len]
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # last_hidden_state 取出最后一层特征。形状: [batch_size, seq_len, 768]
        sequence_output = roberta_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # ---------------------------------------------------------
        # 阶段二：长距离语义交互 (Self-Attention)
        # ---------------------------------------------------------
        # 形状保持不变: [batch_size, seq_len, 768]
        attn_output = self.attention(sequence_output, attention_mask=attention_mask)
        
        # ---------------------------------------------------------
        # 阶段三：句法拓扑特征聚合 (GCN)
        # ---------------------------------------------------------
        # 传入邻接矩阵进行多维矩阵乘法聚合
        # 聚合后形状转换为: [batch_size, seq_len, gcn_out_dim]  (例如从 768 降维到 300)
        gcn_output = self.gcn(attn_output, adj_matrix)
        
        # ---------------------------------------------------------
        # 阶段四：标签空间投影 (Linear)
        # ---------------------------------------------------------
        # 映射到标签数。生成发射得分矩阵。形状: [batch_size, seq_len, num_tags]
        emissions = self.classifier(gcn_output)
        
        # ---------------------------------------------------------
        # 阶段五：根据模式分流 (训练算 Loss / 预测做解码)
        # ---------------------------------------------------------
        if labels is not None:
            # 【训练模式】：调用 CRF 计算负对数似然损失
            # 注意：CRF层需要的 mask 必须和 emissions 形状对齐，直接传入语言模型的 attention_mask 即可
            loss = self.crf(emissions, tags=labels, mask=attention_mask)
            return loss
        else:
            # 【预测模式】：调用 CRF 的维特比算法解码最优路径
            # 返回的是一个嵌套列表，例如 [[0, 0, 5, 6, 0], [0, 1, 2, 0, 0]]
            predictions = self.crf.decode(emissions, mask=attention_mask)
            return predictions

# ================= 司令部整装运行测试 =================
if __name__ == "__main__":
    # 模拟超参数
    BATCH_SIZE = 2
    SEQ_LEN = 12
    NUM_TAGS = 7
    TOKENIZER_NAME = 'roberta-base' # 测试用英文，若网络不通可换成本地路径
    
    # 1. 实例化主模型
    model = ABSAMainModel(model_name_or_path=TOKENIZER_NAME, num_tags=NUM_TAGS, gcn_out_dim=300)
    
    # 2. 伪造输入张量 (完全模拟我们从 Dataset.__getitem__ 拼出的 Batch 数据)
    dummy_input_ids = torch.randint(10, 5000, (BATCH_SIZE, SEQ_LEN))
    dummy_attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], # 第一句话长 10，后 2 个是 PAD
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 第二句话长 8，后 4 个是 PAD
    ], dtype=torch.long)
    
    dummy_adj_matrix = torch.randn(BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    dummy_labels = torch.randint(0, NUM_TAGS, (BATCH_SIZE, SEQ_LEN))
    
    print("\n====== 模拟训练模式 (Forward with Labels) ======")
    model.train() # 切换为训练模式，激活 Dropout
    loss = model(
        input_ids=dummy_input_ids, 
        attention_mask=dummy_attention_mask, 
        adj_matrix=dummy_adj_matrix, 
        labels=dummy_labels
    )
    print(f"训练阶段前向传播成功！算出的批次平均 CRF Loss: {loss.item():.4f}")
    
    print("\n====== 模拟预测模式 (Forward without Labels) ======")
    model.eval() # 切换为评估模式，关闭 Dropout
    with torch.no_grad(): # 关闭梯度上下文，节省显存
        preds = model(
            input_ids=dummy_input_ids, 
            attention_mask=dummy_attention_mask, 
            adj_matrix=dummy_adj_matrix, 
            labels=None # 不传标签
        )
    print(f"预测阶段前向传播成功！")
    print(f"第一句话的预测路径长度 (应为 10): {len(preds[0])} -> {preds[0]}")
    print(f"第二句话的预测路径长度 (应为 8): {len(preds[1])} -> {preds[1]}")
    
    print("\n🎉 宏伟蓝图组装完毕！从数据输入到 Loss 产出/解码输出的端到端大管道已完全打通！")