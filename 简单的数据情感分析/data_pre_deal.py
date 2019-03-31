
# -*- coding: utf-8 -*-

# 数据预处理
# 
# 数据是2500个大众点评中的甜品店评论，包含字段：、评分、评论内容、口味、环境、服务


#引入库
import pandas as pd
import re

data = pd.read_csv('tianpin.csv')


# ### 数据概要
# 
# 查看数据大小以及基础信息
data.info()

# 1.**去除非文本数据：**可以看出，爬虫获取的数据非常多类似“\xa0”的非文本数据，而且都还有一些无意义的干扰数据，

comment = data.loc[:, ['cus_comment']]     #复制cus_comment列的数据，便于后面使用
star = data.loc[:, ['comment_star']]       #复制comment_star列的

#除去非文本数据和无意义文本，只保留中文字符
for index in range(len(data['cus_comment'])):
    line = data['cus_comment'][index].strip().decode('utf-8', 'ignore')  # 处理前进行相关的处理，包括转换成Unicode等
    p2 = re.compile(ur'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    zh = " ".join(p2.split(line)).strip()
    zh = ",".join(zh.split())
    comment['cus_comment'][index] = zh




# 2.**中文分词：**中文文本数据处理，怎么能离开中文分词呢，我们使用jieba库，简单又好用。这里我们把文本字符串处理为以空格区隔的分词字符串
#中文分词
import jieba

def stopwordslist():
    stopwords = [line.strip() for line in open('stopwords.txt').readlines()]
    return stopwords

# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word.encode('utf-8') not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

#对评论进行分词和去停用词
for index in range(len(comment['cus_comment'])):
    comment['cus_comment'][index] = seg_depart(comment['cus_comment'][index])

#合并评论打星列和评论文本列，用于后面文本挖掘和情感分析
result = pd.concat([star, comment], axis=1)

#导出数据
result.to_csv('data.csv', index = 0)

