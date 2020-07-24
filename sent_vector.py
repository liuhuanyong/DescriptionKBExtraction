from albert_zh.extract_feature import BertVector
from tqdm import tqdm
MAX_SEQ_LEN = 200
import numpy as np
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def sim_text(s1, s2):
    sent_vec1 = bert_model.encode([s1])["encodes"][0].mean(axis=1)
    sent_vec2 = bert_model.encode([s2])["encodes"][0].mean(axis=1)
    return cos_sim(sent_vec1, sent_vec2)

while 1:
    s1 = input('enter sent1:').strip()
    s2 = input('enter sent2:').strip()
    print('sim score', sim_text(s1, s2))
