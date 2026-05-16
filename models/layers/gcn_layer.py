import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        """
        初始化单层图卷积网络 (Graph Convolutional Network Layer)
        
        数学公式: H^(l+1) = ReLU(A * H^(l) * W + b)
        
        :param in_features: 输入特征的维度大小 (例如 RoBERTa-base 输出维度通常是 768)
        :param out_features: 经过 GCN 映射后的输出特征维度大小
        :param dropout_rate: 防止模型过拟合的丢弃率
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. 定义可学习的权重矩阵 W 和偏置向量 b
        # 使用 nn.Parameter 包装，将其注册为模型的参数，以便在反向传播时自动计算梯度
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # 2. 定义 Dropout 层，随机丢弃部分神经元，增强模型鲁棒性
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化策略：
        使用 Xavier 均匀分布初始化权重 W，将偏置 b 初始化为 0。
        这能有效避免深度网络在训练初期的梯度消失或梯度爆炸问题。
        """
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, text_features, adj_matrix):
        """
        前向传播逻辑
        
        :param text_features: 文本特征矩阵 H。
                              形状: [batch_size, seq_len, in_features]
        :param adj_matrix: 句法依赖邻接矩阵 A。
                           形状: [batch_size, seq_len, seq_len]
        :return: 图聚合后的新特征矩阵。
                 形状: [batch_size, seq_len, out_features]
        """
        
        # ---------------------------------------------------------
        # 步骤 1：特征投影 (Feature Projection) --> H * W
        # ---------------------------------------------------------
        # text_features: [batch_size, seq_len, in_features]
        # self.weight: [in_features, out_features]
        # PyTorch 的 matmul 非常智能，它会自动在 seq_len 维度上对每一个 Token 执行线性映射
        # 映射后的 support 形状: [batch_size, seq_len, out_features]
        support = torch.matmul(text_features, self.weight)
        
        # ---------------------------------------------------------
        # 步骤 2：图消息传递与特征聚合 (Message Passing & Aggregation) --> A * (H * W)
        # ---------------------------------------------------------
        # adj_matrix: [batch_size, seq_len, seq_len]
        # support: [batch_size, seq_len, out_features]
        # 这里进行的是批量矩阵乘法 (BMM)。
        # 直观理解：邻接矩阵 A 描述了词与词之间的修饰关系。相乘的过程，
        # 就是让句子中的每一个词，根据 A 的连接权重，把周围修饰词的特征 support 吸收过来。
        # 聚合后的 output 形状: [batch_size, seq_len, out_features]
        output = torch.matmul(adj_matrix, support)
        
        # ---------------------------------------------------------
        # 步骤 3：加上偏置项 --> (A * H * W) + b
        # ---------------------------------------------------------
        # output: [batch_size, seq_len, out_features]
        # self.bias: [out_features]
        # 借助 PyTorch 的广播机制 (Broadcasting)，偏置会被自动加到每一个 Token 的特征上
        output = output + self.bias
        
        # ---------------------------------------------------------
        # 步骤 4：非线性激活与正则化 --> ReLU(...) -> Dropout(...)
        # ---------------------------------------------------------
        output = torch.relu(output)
        output = self.dropout(output)
        
        return output

# ================= 测试模块 =================
# 你可以直接运行这个文件来测试这个层是否能正常跑通，且不报错
if __name__ == "__main__":
    # 模拟超参数
    batch_size = 4
    seq_len = 32
    in_features = 768  # RoBERTa 的默认特征维度
    out_features = 300 # 我们希望 GCN 提取出的核心特征维度
    
    # 实例化 GCN 层
    gcn_layer = GCNLayer(in_features=in_features, out_features=out_features)
    
    # 伪造输入数据 (随机生成)
    # 1. 模拟 RoBERTa 输出的动态词向量
    dummy_text_features = torch.randn(batch_size, seq_len, in_features)
    
    # 2. 模拟从 spaCy 提取出的句法邻接矩阵
    # 通常邻接矩阵是一个稀疏的 0/1 矩阵，这里用随机数模拟
    dummy_adj_matrix = torch.randn(batch_size, seq_len, seq_len)
    
    print("====== GCN 层输入维度 ======")
    print(f"文本特征维度 (H): {dummy_text_features.shape}")
    print(f"邻接矩阵维度 (A): {dummy_adj_matrix.shape}")
    
    # 前向传播
    output_features = gcn_layer(dummy_text_features, dummy_adj_matrix)
    
    print("\n====== GCN 层输出维度 ======")
    print(f"聚合后特征维度: {output_features.shape}")
    
    assert output_features.shape == (batch_size, seq_len, out_features), "输出维度与预期不符！"
    print("\n🎉 测试通过！GCN 层维度推导完全正确！")