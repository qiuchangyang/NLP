# utils/metrics.py
from sklearn.metrics import f1_score, classification_report

def compute_f1_score(true_labels, pred_labels, average='macro'):
    """
    计算 F1 分数
    :param true_labels: 真实标签的一维列表
    :param pred_labels: 预测标签的一维列表
    :param average: 'macro' 宏平均，适合类别不平衡的序列标注
    :return: f1 分数 (float)
    """
    return f1_score(true_labels, pred_labels, average=average)

def print_detailed_report(true_labels, pred_labels):
    """
    打印详细的分类报告 (召回率、精确率等)
    """
    print(classification_report(true_labels, pred_labels, zero_division=0))