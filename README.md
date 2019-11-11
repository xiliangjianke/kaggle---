# kaggle-Tatannic
# 导入库
用到的库：
- numpy,pandas,matplotlib,seaborn,xgboost

# 1.导入数据
将测试集与训练集合并，并添加字段0/1用于区分。
此处用到的方法有：

- pd.read.csv:读取数据

- pd.append：将训练集和验证集合并

- pd.set_index:设置行索引名称

# 2.查看数据整体

data.head()

共11个字段名分别为：年龄、船舱号、登船港口、票价、名字、父母子女数、兄弟姐妹数、经济水平、性别、票号。

- 登船港口：C/Q/S三个港口
- 经济地位：分为low/middle/high

其中，Survived/SibSp/Parch/Age/Fare为数值型数据，直观上讲可能会对存活率有影响，可以先对其进行相关性分析，看变量间是否有关联以便筛选。
Embarked/sex为分类变量，后面可尝试进行编码处理

# 3.查看数值型变量间的相关性，用于筛选变量

data["字段名"].corr()求取相关系数，sns.heatmap画出相关系数图

- 相关系数：度量两个变量之间的线性相关程度。

- 独立与相关的关系：独立一定不相关，不相关不一定独立，独立描述两个变量是否有关系，相关描述是否存在线性关系，没有线性关系不代表没有别的关系。只有在二维正态分布时独立和不相关才是充要条件。

## 结论：各变量间无显著相关性。

# 4.查看不同数值变量与存活与否的关系
``` python3
def compare_numerical_to_survive(data, variable, live='Survived'):
result = data[[variable, live]][data[live].notnull()].groupby([variable], as_index=False).mean().sort_values(by=live, ascending=False)
```
其中data[[variable, live]][data[live].notnull()]选取两列且存或为空，groupby按字段进行分组并按均值方法进行聚合，之后sort_values进行排序。

- 父母子女数量
为3的存活率最高，且存活数主要集中在0、1、2、3数量的区间，4和6存活率为0,5的存活率最低，可能是不抛弃不放弃，结果最后葫芦娃救爷爷？

- 兄弟姐妹数量
为1的存活率最高，且随着个数增加，存活率有下降的趋势，同葫芦娃救爷爷。

- 票价
从柱状图中可看出各区间各区间票价存活率无明显区别，都有高有低。

- 年龄
年龄较小和较大存活率都比较高，中间段反而低，给尊老爱幼点个赞！

# 5.查看分类变量与存活的关系
- 性别
女性存活率为0.74，男性仅为0.19，给绅士风度点赞！

- 经济地位
经济地位越高存活率越高，有钱人聪明。

- 登录港口
C港口存活率为0.55最高，其他两个港口相差不大在0.35左右，C港口是富人专用港口？或是女性专用？

# 6.特征工程

## 缺失值填充
data.isnull.sum()
年龄有263个缺失，船舱号有1014个，港口有2个，票价有1个。

对于缺失值比较小的，采用众数和均值进行填充。缺失值多的单独进行处理。

- 船舱号
data.Cabin.value_counts()
可以看出船舱是以字母+数字的方式命名的，因此可以用Unkonwn来替换缺失值，并提取所有船舱的首字母从新命名
``` python
data.Cabin = data.Cabin.fillna("Unknown")

data['Cabin'] = data['Cabin'].str[0]
```
资本家和穷光蛋一般不会乘坐统一船舱，按经济地位分组看看船舱号
``` python
data.groupby('Pclass').Cabin.value_counts()
```
发现不同经济地位中均有补充的船舱号U，但不同经济地位缺失的船舱号应当有**不同的含义**，考虑用*不同经济地位*中的*其它船舱号*来替换U。
``` python
data['Cabin'] = np.where((data.Pclass==1) & (data.Cabin=='U'),'C',
                                            np.where((data.Pclass==2) & (data.Cabin=='U'),'D',
                                                                        np.where((data.Pclass==3) & (data.Cabin=='U'),'G',
                                                                                                    np.where(data.Cabin=='T','C',data.Cabin))))
```

这里替换规则如下：经济地位高，用C替换U；经济地位中，用D替换U，经济地位低，用G替换U。
numpy.where(condition,x,y)，为三运表达式**if conditon x else y**的向量化版本

- 年龄
可以发现，名字字段命中带有称谓，如Mr,Mrs,可以以此作为年龄填充的参考,对于不能作参考的用others来表示
``` python
#获取称谓
data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)

#称谓替换
data['Title'] = np.where((data.Title=='Capt') | (data.Title=='Countess') | (data.Title=='Don') | (data.Title=='Dona')
                        | (data.Title=='Jonkheer') | (data.Title=='Lady') | (data.Title=='Sir') | (data.Title=='Major') | (data.Title=='Rev') | (data.Title=='Col'),'Other',data.Title)

data['Title'] = data['Title'].replace('Ms','Miss')
data['Title'] = data['Title'].replace('Mlle','Miss')
data['Title'] = data['Title'].replace('Mme','Mrs')

```
接下来查看称谓和年龄的关系，并用均值进行填充。
``` python
#获取不同称谓的年龄均值
data.groupby('Title').Age.mean()

#开始填充
data['Age'] = np.where((data.Age.isnull()) & (data.Title=='Master'),5,
                        np.where((data.Age.isnull()) & (data.Title=='Miss'),22,
                                 np.where((data.Age.isnull()) & (data.Title=='Mr'),32,
                                          np.where((data.Age.isnull()) & (data.Title=='Mrs'),37,
                                                  np.where((data.Age.isnull()) & (data.Title=='Other'),45,
                                                           np.where((data.Age.isnull()) & (data.Title=='Dr'),44,data.Age))))))  
```

## 增加新特征
根据之前不同变量和存活与否的关系，打算增加以下变量：
- FamilySize

和兄弟姐妹、父母子女有关，描述家庭的大小
``` python
data['FamilySize'] = data.SibSp + data.Parch + 1
```
- Mother 

拥有Mrs称谓，且有子女，有孩子的母亲会显著影响孩子和她自身的存活率
``` python3
data['Mother'] = np.where((data.Title=='Mrs') & (data.Parch >0),1,0)
```
- Free

那些免费获得船票的乘客，这些乘客也很聪明，可以单独考虑
``` python3
data['Free'] = np.where(data['Fare']==0, 1,0)
```
- TypeOfTicke

船票编号，前缀相同会在同一船舱
代码段较长，主要是编号中字符串的处理，具体可参考源码。

## 编码
data = pd.get_dummies(data)
get_dummpies一步到位！

# 7.模型建立与验证
时间原因，参数是参考别人的没有自己调整，调整的话可以考虑网格搜索+交叉验证，sklearn中有对应API。
分别采用决策树、随机森林和XGB进行建模。

- 数据分割
``` python
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(data[data.Survived.isnull()==False].drop('Survived',axis=1),data.Survived[data.Survived.isnull()==False],test_size=0.30, random_state=2019)
```
利用Result存放结果:
``` python
Results = pd.DataFrame({'Model': [],'Accuracy Score': []})
```
Results = pd.DataFrame({'Model': [],'Accuracy Score': []})

## 决策树
``` python
from sklearn.tree import DecisionTreeClassifier

#max_depth为初始树深度，控制大小防止过拟合。
model = DecisionTreeClassifier(max_depth=4)

#训练模型
model.fit(trainX, trainY)

#数预测
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score

#采用分类准确率分数进行评分
res = pd.DataFrame({"Model":['DecisionTreeClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
Results = Results.append(res)
```

## 随机森林
``` python
from sklearn.ensemble import RandomForestClassifier

#n_estimators为决策树个数，不能太小否则容易欠拟合，过大提升有限但训练时间显著增大
model = RandomForestClassifier(n_estimators=2500, max_depth=4)

#训练
model.fit(trainX, trainY)

#预测
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score

#评分
res = pd.DataFrame({"Model":['RandomForestClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
Results = Results.append(res)
```

## XGBoost
``` python
from xgboost.sklearn import XGBClassifier

#参数解释，learning_rate学习速率，min_child_weight最小叶子节点样本权重和；gamma分裂最小损失函数；
subsample随机采样比例；colsample_bytree控制每棵树随即采用的特征的占比；scale_pos_weight类别样本十分不均衡时设定为正值加快收敛；
model = XGBClassifier(learning_rate=0.001,n_estimators=2500,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27,
                                reg_alpha=0.00006)
model.fit(trainX, trainY)
y_pred = model.predict(testX)
from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['XGBClassifier'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})
Results = Results.append(res)
```
# 8.最终模型分数
模型|分数
-|-
决策树|0.832
随机森林|0.84
XGB|0.85
XGB 牛X!
