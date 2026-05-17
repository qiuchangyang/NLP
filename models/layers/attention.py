import torch
import torch.nn as nn
import math

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout_rate=0.1):
        """
        初始化多头自注意力层 (Multi-Head Self-Attention)
        
        :param hidden_size: 输入特征的维度 (必须能被 num_heads 整除，如 768 / 8 = 96)
        :param num_heads: 注意力头的数量。多头机制允许模型同时关注不同子空间的信息。
        :param dropout_rate: 防止过拟合的丢弃率
        """
        super(SelfAttentionLayer, self).__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除！")
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # 每个注意力头负责的特征维度 (例如 768 / 8 = 96)
        self.head_dim = hidden_size // num_heads 
        
        # 定义 Q, K, V 的线性映射层 (招聘三个员工，分别负责生成 Query, Key, Value)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # 输出的线性映射层 (把多头拼接后的结果再融合一下)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        
        :param hidden_states: 输入特征 (比如 GCN 输出的特征)。形状: [batch_size, seq_len, hidden_size]
        :param attention_mask: 掩码 (用来遮蔽 PAD token)。形状: [batch_size, seq_len]
        :return: 经过注意力加权后的新特征。形状: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # ---------------------------------------------------------
        # 步骤 1：生成 Q, K, V 并进行多头切分 (Dimension Magic)
        # ---------------------------------------------------------
        # 映射后的形状仍为 [batch_size, seq_len, hidden_size]
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        # 拆分多头：将 hidden_size 拆成 (num_heads, head_dim)
        # 形状变为: [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 维度交换：为了让矩阵乘法发生在 seq_len 和 head_dim 之间，把 num_heads 移到前面
        # 形状变为: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # ---------------------------------------------------------
        # 步骤 2：计算注意力打分 (Attention Scores) -> Q * K^T / sqrt(d)
        # ---------------------------------------------------------
        # K.transpose(-1, -2) 把最后两个维度倒过来，变成 [batch_size, num_heads, head_dim, seq_len]
        # Q 和 K^T 相乘，得到注意力打分矩阵 scores
        # 形状: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2))
        
        # 缩放 (Scaling)：除以根号下 head_dim，防止内积过大导致 Softmax 梯度消失
        scores = scores / math.sqrt(self.head_dim)
        
        # 处理 Mask：把 PAD 的位置的打分变成负无穷大 (-1e9)
        # 这样 Softmax 之后，这些位置的概率就会变成 0，模型就不会去关注没用的废料了
        if attention_mask is not None:
            # attention_mask 原本是 [batch_size, seq_len]，扩展成 [batch_size, 1, 1, seq_len] 以匹配 scores 的形状
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # mask 中为 0 的地方 (PAD)，替换为 -1e9
            scores = scores.masked_fill(extended_mask == 0, -1e9)
            
        # 归一化：将打分变成总和为 1 的概率分布
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # ---------------------------------------------------------
        # 步骤 3：根据概率加权求和获取新特征 -> Weights * V
        # ---------------------------------------------------------
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        # V: [batch_size, num_heads, seq_len, head_dim]
        # 结果 context 形状: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, V)
        
        # ---------------------------------------------------------
        # 步骤 4：拼接多头并输出
        # ---------------------------------------------------------
        # 先把 num_heads 换回原来的位置: [batch_size, seq_len, num_heads, head_dim]
        # contiguous() 是因为 transpose 弄乱了内存连续性，必须重新排列内存才能用 view
        context = context.transpose(1, 2).contiguous()
        
        # 把后两个维度拍扁，拼回 hidden_size: [batch_size, seq_len, hidden_size]
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        # 经过最后一层线性映射融合
        output = self.out_proj(context)
        
        return output

# ================= 测试模块 =================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    hidden_size = 768
    num_heads = 8
    
    # 实例化自注意力层
    attn_layer = SelfAttentionLayer(hidden_size=hidden_size, num_heads=num_heads)
    
    # 伪造输入特征 (假设这是 GCN 提完特征后传过来的数据)
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    
    # 伪造 Mask (第一句话全有，第二句话只有前 10 个词有效)
    dummy_mask = torch.ones(batch_size, seq_len)
    dummy_mask[1, 10:] = 0
    
    print("====== Attention 层输入维度 ======")
    print(f"输入特征维度: {dummy_input.shape}")
    print(f"Mask 维度: {dummy_mask.shape}")
    
    # 前向传播
    output = attn_layer(dummy_input, attention_mask=dummy_mask)
    
    print("\n====== Attention 层输出维度 ======")
    print(f"输出特征维度: {output.shape}")
    
    assert output.shape == (batch_size, seq_len, hidden_size), "输出维度错误！"
    print("\n🎉 测试通过！Attention 层的维度推演如丝般顺滑！")