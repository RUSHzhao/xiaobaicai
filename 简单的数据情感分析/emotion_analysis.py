
# -*- coding: utf-8 -*-

# # 大众点评评价情感分析
# 模型的效果还可以的样子，yeah~接下来我们好好讲讲怎么做的哈，我们通过爬虫爬取了大众点评广州3家最热门甜品店店的2500+条评论信息以及评分作为训练数据，前面的分析我们得知*样本很不均衡*。


import pandas as pd
from matplotlib import pyplot as plt
import jieba
data = pd.read_csv('data.csv')     #读取预处理后的程序
data.head()


# ### 构建标签值
# 大众点评的评分（星级）分为10-50分，分别代表一颗星~五颗星
# 因此我们把10-20记为0（差评）,4-5记为1（好评）
# 3为中评，对我们的情感分析作用不大，丢弃掉这部分数据，但是可以作为训练语料模型的语料。
# 我们的情感评分可以转化为标签值为1的概率值，这样我们就把情感分析问题转为文本分类问题了。


#构建label值
def zhuanhuan(score):
    if score > 30:
        return 1
    elif score < 30:
        return 0
    else:
        return None
    
#标签值转换
data['target'] = data['comment_star'].map(lambda x:zhuanhuan(x))


# ### 文本特征处理
# 
# 中文文本特征处理，需要进行中文分词，jieba分词库简单好用。
# 接下来需要过滤停用词，网上能够搜到现成的。最后就要进行文本转向量
# 这里我们使用sklearn库的TF-IDF工具进行文本特征提取。

from sklearn.model_selection import train_test_split

#将gbk编码的字符串变为utf-8编码，以便之后能够操作
star = data.loc[:, ['target']]
comment = data.loc[:, ['cus_comment']]

for i in range(len(data['cus_comment'])):
    s = comment['cus_comment'][i].decode('gbk')
    comment['cus_comment'][i] = s.encode('utf-8')

#合并评论正负类别列（0，1列）和评论文本列
result = pd.concat([star, comment], axis=1)
data_model = result.dropna()

#切分测试集、训练集
x_train, x_test, y_train, y_test = \
    train_test_split(data_model['cus_comment'], data_model['target'], random_state=3, test_size=0.25)
print '输出训练集的前5项'
print x_train[:5]

#引入停用词
infile = open("stopwords.txt")
stopwords_lst = infile.readlines()
stopwords = [x.strip() for x in stopwords_lst]

#中文分词
def fenci(train_data):
    words_df = train_data.apply(lambda x:' '.join(jieba.cut(x)))
    return words_df


#使用TF-IDF进行文本转向量处理
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tv.fit(x_train)


# ### 机器学习建模
# 特征和标签已经准备好了，接下来就是建模了。
# 这里我们使用文本分类的经典算法朴素贝叶斯算法，而且朴素贝叶斯算法的计算量较少。
# 特征值是评论文本经过TF-IDF处理的向量，标签值评论的分类共两类，好评是1，差评是0。情感评分为分类器预测分类1的概率值。


#分类效果的准确率
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

classifier = MultinomialNB()
#模型训练
classifier.fit(tv.transform(x_train), y_train)
#使用训练好的模型进行预测，输出准确率
print '分类效果的准确率:'
print classifier.score(tv.transform(x_test), y_test)

#计算分类器的AUC值
y_pred = classifier.predict_proba(tv.transform(x_test))[:,1]
print '分类器的AUC值：'
print roc_auc_score(y_test, y_pred)
print '\n'


#计算一条评论文本的情感评分
def ceshi(model, strings):
    strings_fenci = fenci(pd.Series([strings]))
    return float(model.predict_proba(tv.transform(strings_fenci))[:,1])


#从大众点评网找两条评论来测试一下
test1 = '很好吃，环境好，所有员工的态度都很好，奶茶一如既往的好' #5星好评
test2 = '奶茶不好喝，味道不行，员工服务态度也很差' #1星差评

print('好评实例的模型预测情感得分为{}\n差评实例的模型预测情感得分为{}'.format(ceshi(classifier,test1),ceshi(classifier,test2)))


# 可以看出，准确率和AUC值都非常不错的样子，但实际测试中，5星好评模型预测出来了，1星差评缺预测错误。
#接下来我们查看混淆矩阵
from sklearn.metrics import confusion_matrix
y_predict = classifier.predict(tv.transform(x_test))

#混淆矩阵，可以看出负类x_test被成功预测和被预测为正类类的数量以及正类类y_test被成功预测和预测为负类的数量
cm = confusion_matrix(y_test, y_predict)
print '正负类数据不平均时的混淆矩阵：'
print cm
print '\n'

# 处理样本不均衡问题的方法，首先可以选择调整阈值，使得模型对于较少的类别更为敏感或者选择合适的评估标准，比如ROC或者F1，而不是准确度（accuracy）。
# 另外一种方法就是通过采样（sampling）来调整数据的不平衡。其中欠采样抛弃了大部分正例数据，从而弱化了其影响，可能会造成偏差很大的模型，同时，数据总是宝贵的，抛弃数据是很奢侈的。
# 另外一种是过采样，下面我们就使用过采样方法来调整。
# 
# ### 过采样（单纯复制）
# 
# 单纯的重复了反例，因此会过分强调已有的反例。如果其中部分点标记错误或者是噪音，那么错误也容易被成倍的放大。因此最大的风险就是对反例过拟合。


#输出正面评价和负面评价的数量
print '原始正面评价和负面评价的数量'
print data['target'].value_counts()
print '\n'

#把0类样本复制10次，构造训练集
index_tmp = y_train == 0
y_tmp = y_train[index_tmp]
x_tmp = x_train[index_tmp]
x_train2 = pd.concat([x_train,x_tmp,x_tmp,x_tmp,x_tmp,x_tmp,x_tmp,x_tmp,x_tmp,x_tmp,x_tmp])
y_train2 = pd.concat([y_train,y_tmp,y_tmp,y_tmp,y_tmp,y_tmp,y_tmp,y_tmp,y_tmp,y_tmp,y_tmp])


#使用过采样样本(简单复制)进行模型训练，并查看准确率
clf2 = MultinomialNB()
clf2.fit(tv.transform(x_train2), y_train2)
y_pred2 = clf2.predict_proba(tv.transform(x_test))[:,1]
print '分类器的AUC值：'
print roc_auc_score(y_test, y_pred2)

#查看此时的混淆矩阵，查看准确率
y_predict2 = clf2.predict(tv.transform(x_test))
cm = confusion_matrix(y_test, y_predict2)
print '单纯复制0类样本时的混淆矩阵：'
print cm
print '\n'

# 可以看出，即使是简单粗暴的复制样本来处理样本不平衡问题，负样本的识别率大幅上升了，变为77%，满满的幸福感呀~我们自己写两句评语来看看


print '过采样（单纯复制）测试结果——奶茶不好喝，太甜了。12元一杯不值。：'
print ceshi(clf2, '奶茶不好喝，太甜了。12元一杯不值。')
print '\n'

# 可以看出把0类别的识别出来了，太棒了~

# ### 过采样（SMOTE算法）
# 
# SMOTE（Synthetic minoritye over-sampling technique,SMOTE），是在局部区域通过K-近邻生成了新的反例。
# 相较于简单的过采样，SMOTE降低了过拟合风险，但同时运算开销加大


#使用SMOTE进行样本过采样处理
from imblearn.over_sampling import SMOTE

oversampler = SMOTE(random_state=0)
x_train_vec = tv.transform(x_train)
x_resampled, y_resampled = oversampler.fit_sample(x_train_vec, y_train)


#原始的样本分布
print '原始的样本分布：'
print y_train.value_counts()


#经过SMOTE算法过采样后的样本分布情况
print '经过SMOTE算法过采样后的样本分布情况：'
print pd.Series(y_resampled).value_counts()


# 我们经过插值，把0类数据也丰富到和1类数据一样多了，这时候正负样本的比例为1:1，接下来我们用平衡后的数据进行训练

#使用过采样样本(SMOTE)进行模型训练，并查看准确率
clf3 = MultinomialNB()
clf3.fit(x_resampled, y_resampled)
y_pred3 = clf3.predict_proba(tv.transform(x_test))[:, 1]
print '使用SMOTE进行模型训练的准确率：'
print roc_auc_score(y_test, y_pred3)

#查看此时的混淆矩阵，查看准确率
y_predict3 = clf3.predict(tv.transform(x_test))
#混淆矩阵，可以看出正类x_test被成功预测和被预测为负类的数量以及负类y_test被成功预测和预测为正类的数量
cm = confusion_matrix(y_test, y_predict3)
print '正负类数据平均时的混淆矩阵：'
print cm
print '\n'


#到网上找一条差评来测试一下情感评分的预测效果
test3 = '奶茶不好喝，太甜了。12元一杯不值。'
print '过采样（SMOTE算法）测试结果——奶茶不好喝，太甜了。12元一杯不值。'
print ceshi(clf3, test3)
print '\n'


# 可以看出，使用SMOTE插值与简单的数据复制比起来，AUC率略有提高，实际预测效果也挺好
# ### 模型评估测试

#词向量训练
tv2 = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tv2.fit(data_model['cus_comment'])

#SMOTE算法
X_tmp = tv2.transform(data_model['cus_comment'])
y_tmp = data_model['target']
sm = SMOTE(random_state=0)
X, y = sm.fit_sample(X_tmp, y_tmp)

clf = MultinomialNB()
clf.fit(X, y)

def fenxi(strings):
    strings_fenci = fenci(pd.Series([strings]))
    return float(clf.predict_proba(tv2.transform(strings_fenci))[:,1])


#使用评论进行测试
print '终极测试——糯米外皮不绵滑，豆沙馅也少，没有香甜味。12元一碗不值'
print fenxi('糯米外皮不绵滑，豆沙馅很少，没有香甜味。12元一碗不值。')

print '终极测试——很漂亮，很好吃的蛋糕，下次还会再来'
print fenxi('很漂亮，很好吃的蛋糕，下次还会再来')

print '终极测试——奶茶一般般，没什么特别的'
print fenxi('奶茶一般般，没什么特别的')

print '掐的时间等开门,号,我要半糖正常冰中杯柠檬养乐多,以前喝过,知道有两瓶养乐多和柠檬汁以及柠檬片,看到封盖的时候没有柠檬片问了一句没有柠檬嘛,那女的来句,没准备好,没有准备好你可以提前告知我下,她又语出惊人的来句这也就是装饰的而已,我就来气了,柠檬在那里,请洗干净切片放入,不好意思,你态度好打招呼我,若你这样的鸟样那么我也不会客气'
print fenxi('掐的时间等开门,号,我要半糖正常冰中杯柠檬养乐多,以前喝过,知道有两瓶养乐多和柠檬汁以及柠檬片,看到封盖的时候没有柠檬片问了一句没有柠檬嘛,那女的来句,没准备好,没有准备好你可以提前告知我下,她又语出惊人的来句这也就是装饰的而已,我就来气了,柠檬在那里,请洗干净切片放入,不好意思,你态度好打招呼我,若你这样的鸟样那么我也不会客气')

print '叫过好几次外卖,没有一次是味道是正常的,因为公司附近的一点点只有这一家,所以尝试了好几次,想着总有一次是正常的吧,今天叫了外卖,更夸张,两杯波霸奶茶,颜色不一样也就算了,味道也是差了十万八千里,淡色的那一杯只有奶味和甜味,而且甜到齁,相比之下深色的那杯就好很多,不知道这家店做奶茶的员工是按什么标准来做的,有一次点了个七分甜,苦的跟无糖一样,今天要的正常甜,和别的门店七分甜一样'
print fenxi('叫过好几次外卖,没有一次是味道是正常的,因为公司附近的一点点只有这一家,所以尝试了好几次,想着总有一次是正常的吧,今天叫了外卖,更夸张,两杯波霸奶茶,颜色不一样也就算了,味道也是差了十万八千里,淡色的那一杯只有奶味和甜味,而且甜到齁,相比之下深色的那杯就好很多,不知道这家店做奶茶的员工是按什么标准来做的,有一次点了个七分甜,苦的跟无糖一样,今天要的正常甜,和别的门店七分甜一样')

# ### 我们对于这次基于机器学习的文本挖掘和分析有一代的想法（后续优化方向）
# 
# - 使用更复杂的机器学习模型如神经网络、支持向量机等
# - 模型的调参
# - 行业词库的构建
# - 增加数据量
# - 优化情感分析的算法
# - 增加标签提取等
# - 项目部署到服务器上，更好地分享和测试模型的效果
