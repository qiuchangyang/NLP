import torch
import torch.nn as nn
from torchcrf import CRF

class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        """
        初始化 CRF 层
        :param num_tags: 标签的总数 (比如我们的任务是 7: O, B-NEG, I-NEG, B-NEU, I-NEU, B-POS, I-POS)
        """
        super(CRFLayer, self).__init__()
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, emissions, tags, mask=None):
        """
        【训练模式】计算 CRF 的负对数似然损失 (Negative Log-Likelihood Loss)
        
        :param emissions: 发射矩阵 (也就是底层网络如 GCN 或 Linear 算出来的值)。
                          形状: [batch_size, seq_len, num_tags]
        :param tags: 真实的标签序列 (我们 Dataset 吐出来的 labels)。
                     形状: [batch_size, seq_len]
        :param mask: 注意力掩码 (告诉 CRF 哪些是 [PAD]，不要算它们的 Loss)。
                     形状: [batch_size, seq_len]，类型必须是 bool 或 byte
        :return: 算出的 Loss 标量值
        """
        # 强制将 mask 转为 bool 类型 (因为 RoBERTa 出来的 attention_mask 是 long 类型)
        if mask is not None:
            mask = mask.bool()
            
        # self.crf() 默认计算的是 对数似然 (Log-Likelihood)，这是一个需要最大化的值
        # 但深度学习的优化器都是做“梯度下降” (求最小值)，所以我们要在前面加个负号 '-'
        # reduction='mean' 表示对整个 Batch 的 Loss 取平均
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def decode(self, emissions, mask=None):
        """
        【预测模式】使用维特比算法 (Viterbi) 解码出最优路径
        
        :param emissions: 发射矩阵。形状: [batch_size, seq_len, num_tags]
        :param mask: 注意力掩码。形状: [batch_size, seq_len]
        :return: 预测出的最优标签序列，返回的是一个嵌套列表 List[List[int]]
        """
        if mask is not None:
            mask = mask.bool()
            
        # decode 会自动寻找一条全局概率最大的标签路径
        return self.crf.decode(emissions, mask=mask)

# ================= 测试模块 =================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    num_tags = 7  # 0~6
    
    # 实例化我们的 CRF 层
    crf_layer = CRFLayer(num_tags=num_tags)
    
    # 1. 模拟底层网络 (比如 GCN + Linear) 输出的特征 (未经 Softmax 的原始打分)
    # [batch_size, seq_len, num_tags]
    dummy_emissions = torch.randn(batch_size, seq_len, num_tags)
    
    # 2. 模拟真实的标签 (在 0 到 6 之间随机生成)
    dummy_tags = torch.randint(0, num_tags, (batch_size, seq_len))
    
    # 3. 模拟 mask (假设第一句话长度是 10，第二句话长度是 7，后面 3 个是 PAD)
    dummy_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    ], dtype=torch.uint8)
    
    print("====== 测试训练阶段 (算 Loss) ======")
    # 注意：CRF 自己算 Loss，不需要再到外面去调 CrossEntropyLoss！
    loss = crf_layer(dummy_emissions, dummy_tags, mask=dummy_mask)
    print(f"计算出的 CRF Loss: {loss.item():.4f}")
    
    print("\n====== 测试预测阶段 (维特比解码) ======")
    predictions = crf_layer.decode(dummy_emissions, mask=dummy_mask)
    
    # 你会发现第二句话只输出了 7 个预测结果，因为 mask 把最后 3 个 PAD 过滤掉了！
    print(f"第一句话的预测路径 (长度 {len(predictions[0])}): {predictions[0]}")
    print(f"第二句话的预测路径 (长度 {len(predictions[1])}): {predictions[1]}")
    print("\n🎉 测试通过！CRF 层完美运行！")