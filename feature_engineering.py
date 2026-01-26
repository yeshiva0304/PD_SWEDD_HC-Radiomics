import os
import nibabel as nib
from pathlib import Path

# 数据目录路径
mydir = r'C:\Users\0304\Desktop\PD_HC_SWEDD'  # 替换为实际路径

# 脑区名称列表
modals = [
    "T1-Caudate Nucleus", "T1-Globus Pallidus", "T1-Ventral Midbrain",
    "T1-Nucleus accumbens", "T1-Putamen", "T1-Thalamus",
    "T2-Caudate Nucleus", "T2-Globus Pallidus", "T2-Ventral Midbrain",
    "T2-Nucleus accumbens", "T2-Putamen", "T2-Thalamus"
]

# 存储所有图像和掩模数据
all_images = []
all_masks = []

# 遍历每个脑区，加载图像和掩模数据
for modal in modals:
    images_dir = os.path.join(mydir, modal, 'images')
    masks_dir = os.path.join(mydir, modal, 'masks')

    # 确保文件夹存在
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"图像或掩模文件夹未找到: {modal}")
        continue

    # 获取所有图像和掩模文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # 加载图像和掩模
    for img_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)

        # 使用 nibabel 读取图像和掩模
        image = nib.load(image_path)
        mask = nib.load(mask_path)


        # 将图像和掩模添加到列表中
        all_images.append(image)
        all_masks.append(mask)

    print(f"已加载 {modals} 的 {len(image_files)} 张图像和掩模。")
# 查看已加载的图像数量和掩模数量
print(f"总共加载了 {len(all_images)} 张图像和 {len(all_masks)} 个掩模。")

# 第二段代码：加载标签数据
import pandas as pd
# 假设你的 label.csv 文件在 D:/swedd/label1.csv 
label_file = r'C:\Users\0304\Desktop\PD_HC_SWEDD\label.csv'
label_data = pd.read_csv(label_file)

# 显示标签数据
print(label_data)

#第三段代码
import warnings
from radiomics import featureextractor
from pathlib import Path
from onekey_algo.custom.components.Radiology import diagnose_3d_image_mask_settings,get_image_mask_from_dir
warnings.filterwarnings("ignore")
from onekey_algo.custom.components.Radiology import ConventionalRadiomics
from pathlib import Path
from onekey_algo.custom.components.Radiology import diagnose_3d_image_mask_settings, get_image_mask_from_dir
rad_ = None


for modal in modals:
    if os.path.exists(f'results/rad_features_{modal}.csv'):

        rad_data = pd.read_csv(f'results/rad_features_{modal}.csv', header=0)
    else:
        images, masks = get_image_mask_from_dir(os.path.join(mydir, modal))
        #如果要自定义一些特征提取方式，可以使用param_file
        param_file = r'C:\Users\0304\Desktop\exampleMR_3mm.yaml'
        radiomics=ConventionalRadiomics(param_file,correctMask=True)
        radiomics.extract(images,masks)
        rad_data = radiomics.get_label_data_frame(label=1)
        rad_data.columns=[f"{c.replace('-','_')}_{modal}"if c != 'ID' else 'ID' for c in rad_data.columns]
    rad_data.to_csv(f'results/rad_features_{modal}.csv', header=True,index=False)
    if not os.path.exists('results'):
        os.makedirs('results')
    if rad_ is None:
        rad_ = rad_data
    else:
        rad_ = pd.merge(rad_, rad_data, on='ID', how='inner')
rad_data= rad_
rad_data
#处理和保存结果
print(f"特征提取完成：{rad_data}")
#第四段：特征统计
import matplotlib.pyplot as plt
from IPython.display import display
sorted_counts = pd.Series([c.split('_')[-3] for c in rad_data.columns if c != 'ID']).value_counts()
sorted_counts=pd.DataFrame(sorted_counts,columns=['count']).reset_index()
sorted_counts=sorted_counts.sort_values('count')
display(sorted_counts)
plt.pie(sorted_counts['count'],labels=[i for i in sorted_counts['index']],startangle=0,counterclock=False,autopct='%1.1f%%'
        ,textprops={'fontweight': 'bold', 'fontsize': 18})
plt.savefig(f'img/Rad_feature_radio.png',bbox_inches='tight')
plt.show()
#第五段：标注数据（数据是以csv格式进行存储，这里如果是其他格式，可以使用自定义函数读取出每个样本的结果，该代码的目的是确保 ID 列中的所有文件名都以 .nii.gz 后缀结束。通过 .map() 和 lambda 函数，它检查每个文件名并为缺少 .nii.gz 后缀的文件添加此后缀）
label_data=pd.read_csv(label_file)
label_data['ID'] = label_data['ID'].map(lambda x: f"{x}.nii.gz" if not x.endswith('.nii.gz') else x)
label_data.head()
#第六段：将label_data与rad_data进行合并
combined_data = pd.merge(rad_data, label_data, on='ID', how='inner')
ids=combined_data['ID']
combined_data=combined_data.drop(['ID'],axis=1)
# Define the label columns you want to count, for example:
labels = [col for col in label_data.columns if col != 'ID']
print(combined_data[labels].value_counts())
combined_data.columns
combined_data.describe()
#z—score标准化
from onekey_algo.custom.components.comp1 import normalize_df
data=normalize_df(combined_data,not_norm=labels)
data=data.dropna(axis=1)
data.describe()
#划分数据集
import onekey_algo.custom.components as okcomp
from sklearn.model_selection import train_test_split
print("正在划分数据集...")
y_data = data[labels]
X_data = data.drop(labels, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42,stratify=y_data[labels[0]]
    )
print(f"数据集划分完成：训练集样本数 {X_train.shape}, 测试集样本数 {X_test.shape}")

#统计检验
import seaborn as sns
from onekey_algo.custom.components.stats import clinic_stats
print("\n步骤 1: 在训练集上进行P值筛选...")
train_data_for_stats = pd.concat([X_train, y_train], axis=1)
stats_df=clinic_stats(train_data_for_stats,stats_columns=list(X_train.columns[0:-1]),label_column=labels[0],continuous_columns=list(X_train.columns))

#输出特征分布的图
import matplotlib.pyplot as plt
  # 定义 map2float 函数，将字符串转换为浮动类型
def map2float(x):
    try:
        return float(str(x)[1:])
    except:
        return 1
 
  # 转换 'pvalue' 列的数据为 float 类型
stats_df['pvalue'] = stats_df['pvalue'].apply(map2float)
  # 处理 'group' 列，通过提取 'feature_name' 列中的最后一个元素来填充 'group' 列
stats_df['group'] = stats_df['feature_name'].apply(lambda x: x.split('_')[-3])
  # 选择需要的列
stats_df_plot = stats_df[['feature_name', 'pvalue', 'group']]
  # 使用 Seaborn 绘制小提琴图
g = sns.catplot(x="group", y="pvalue", data=stats_df_plot, kind="violin")
  # 设置图表大小
g.fig.set_size_inches(15, 10)
  # 使用 seaborn 绘制带有黑色轮廓的图形
sns.stripplot(x="group", y="pvalue", data=stats_df_plot, ax=g.ax, color='black')
  # 保存图表为 PNG 格式
plt.savefig(f'img/Rad_feature_stats.png', bbox_inches='tight')
plt.close()
#调整p值进行筛选
pvalue=0.05
p_selected_features=list(stats_df[stats_df['pvalue'] < pvalue]['feature_name'])
print(f"P值筛选后的特征数量：{len(p_selected_features)}")

# 将筛选结果应用到训练集和测试集
X_train_p = X_train[p_selected_features]
X_test_p = X_test[p_selected_features]
print(f"P值筛选后：训练集维度 {X_train_p.shape}, 测试集维度 {X_test_p.shape}")
#2.相关系数筛选（①pearson 相关系数，②spearman 相关系数，③kendall相关系数）
from onekey_algo.custom.components.comp1 import select_feature
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
print("\n步骤 2: 在P值筛选后的训练集上进行相关系数筛选...")
pearson_corr = X_train_p.corr('pearson')
# 根据相关系数，对于相关性比较高的特征（一般文献取corr>0.9）两者保留其一
corr_selected_features = select_feature(pearson_corr, threshold=0.9, topn=128, verbose=True)
print(f"相关系数筛选后的特征数量：{len(corr_selected_features)}")

# 将相关系数筛选结果应用到训练集和测试集
X_train_corr = X_train_p[corr_selected_features]
X_test_corr = X_test_p[corr_selected_features]
print(f"相关性筛选后：训练集维度 {X_train_corr.shape}, 测试集维度 {X_test_corr.shape}")

#相关系数可视化,通过修改变量名，可视化不同相关系数下的相关矩阵
# 注意：当特征特别多的时候大于100，尽量不要可视化，否则运行时间会特别长
#import seaborn as sns
#import onekey_algo.custom.components.compl import draw_matrix
#if data.shape[1] < 100:
#    plt.figure(figsize=(50.0, 40.0))
#选择可视化的相关系数
#draw_matrix(pearson_corr,annot=True,cmap='Y1GnBu',cbar=False)
#plt.savefig(f'img/Rad_feature_pearson_corr.svg', bbox_inches='tight')
#相关系数可视化，通过修改变量名，可视化不同相关系数下的相聚类分析矩阵
#注意：当特征特别多的时候大于100，尽量不要可视化，否则运行时间会特别长
#import matplotlib.pyplot as plt
# 假设 'data' 是您的数据框，'pearson_corr' 是相关系数矩阵
#if data.shape[1] < 100:
    # 绘制相关系数矩阵的聚类热图
    #pp = sns.clustermap(pearson_corr, linewidths=0.5, figsize=(50.0, 40.0), cmap='YlGnBu')
    # 设置y轴刻度标签的旋转角度为0
    #plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)
    # 保存图像为SVG格式
    #plt.savefig('img/Rad_feature_cluster.svg', bbox_inches='tight')
#3.mRMR筛选
print("\n步骤 3: 在P值筛选后的训练集上进行mRMR筛选...")

# 定义 mRMR 特征选择函数
def mrmr(X, y, num_features=10):
    """
    使用最小冗余最大相关性（mRMR）方法选择特征。
    
    参数:
    X -- 特征数据（矩阵）
    y -- 标签
    num_features -- 选择的特征数目
    
    返回:
    selected_features -- 选择的特征
    """
    if isinstance(y, pd.DataFrame):
       y= y.iloc[:, 0]  # 如果y是DataFrame，取第一列作为目标变量
   #将目标变量y编码成数字，如果y是分类变量
    if y.dtype == 'object':
        le = LabelEncoder()       
        y = le.fit_transform(y)

    # 计算每个特征与目标变量的互信息
    mi_with_target = mutual_info_regression(X, y)
# 初始化空列表用于存储选择的特征
    selected_features_indices = []
    feature_indices= list(range(X.shape[1]))
    if not feature_indices:
        return X.columns[[]]
    # 选择第一个特征：与目标变量互信息最大的特征
    first_feature_idx = np.argmax(mi_with_target)
    selected_features_indices.append(first_feature_idx)
    # 计算特征之间的冗余（即每对特征之间的互信息）
    
    mi_between_features = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            mi = mutual_info_regression(X.iloc[:, [i]], X.iloc[:, j].squeeze())[0]
            mi_between_features[i, j] = mi
            mi_between_features[j, i] = mi

    # 选择特征
    for _ in range(1,min(num_features,X.shape[1])):
        best_score = -np.inf
        best_feature_idx = -1 
        candidate_indices = [idx for idx in feature_indices if idx not in selected_features_indices]

        for feature_idx in candidate_indices:
            relevance = mi_with_target[feature_idx]
            redundancy = np.mean([mi_between_features[feature_idx, selected_idx] for selected_idx in selected_features_indices])
            score = relevance - redundancy
            
            if score > best_score:
                best_score = score
                best_feature_idx = feature_idx
        
        if best_feature_idx != -1:
            selected_features_indices.append(best_feature_idx)
        else:
            break

    return X.columns[selected_features_indices]

# 在经过相关性筛选的训练集上运行 mRMR，选择20个特征
mrmr_selected_features = mrmr(X_train_corr, y_train, num_features=30)
print(f"mRMR选择的特征数量: {len(mrmr_selected_features)}")
print("mRMR选择的特征:", mrmr_selected_features.tolist())


# 将筛选结果应用到训练集和测试集
X_train_mrmr = X_train_corr[mrmr_selected_features]
X_test_mrmr = X_test_corr[mrmr_selected_features]
print(f"mRMR筛选后：训练集维度 {X_train_mrmr.shape}, 测试集维度 {X_test_mrmr.shape}")


#4.LASSO回归
from onekey_algo.custom.components.comp1 import lasso_cv_coefs
column_names = X_train_mrmr.columns
alpha = lasso_cv_coefs(X_train_mrmr, y_train, column_names=column_names)
plt.figure(figsize=(8, 6))  # 调整为合适大小
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig(f'img/Rad_feature_lasso.svg', bbox_inches='tight')
plt.show()
#LASSO模型效能
okcomp.comp1.lasso_cv_efficiency(X_train_mrmr,y_train,points=50)
plt.figure(figsize=(8, 6))  # 调整为合适大小
plt.xticks(rotation=45, fontsize=8)  # 旋转45度，字体变小
plt.tight_layout()#调整布局，避免文字重叠
plt.savefig(f'img/Rad_feature_MSE.svg', bbox_inches='tight')
plt.show()

#惩罚系数（使用交叉验证的惩罚系数作为模型训练的基础）
from sklearn import linear_model
models=[]
for label in labels:
    clf=linear_model.Lasso(alpha=alpha)
    clf.fit(X_train_mrmr, y_train[label])
    models.append(clf)

#特征筛选（筛选出其中coef>0的特征并打印出相应的公式）
COEF_THRESHOLD = 1e-6  # 筛选的特征阈值
scores = []  # 用于存储公式
lasso_final_features_list = []  # 存储符合阈值的特征
feat_coef= []  # 存储特征名和系数的列表

for label, model in zip(labels, models):
    # 获取特征名和模型系数
    feat_coef = list(zip(column_names, model.coef_))
    # 选择符合阈值的特征
    selected_coefs = [(feat, coef) for feat, coef in feat_coef if abs(coef) > COEF_THRESHOLD]
    lasso_final_features_list.append([feat for feat, _ in selected_coefs])
    # 构建Lasso回归公式
    formula = ' + '.join([f"{coef:+.6f} * {feat_name}" for feat_name, coef in selected_coefs])
    # 创建每个模型的公式
    score = f"{label} = {model.intercept_} {'+' if not formula or formula.strip().startswith('-') else '+'} {formula}"
    scores.append(score)
# 输出第一个模型的公式
print("\nLASSO回归模型公式:")
print(scores[0])

# 根据LASSO筛选的特征，创建最终的数据集
final_features_from_lasso = lasso_final_features_list[0] if lasso_final_features_list else []
X_train_final = X_train_mrmr[final_features_from_lasso]
X_test_final = X_test_mrmr[final_features_from_lasso]
print(f"LASSO筛选后：训练集维度 {X_train_final.shape}, 测试集维度 {X_test_final.shape}")

# (可选) 样本可视化：在最终选择的特征空间上
from onekey_algo.custom.components.comp1 import analysis_features
print("\n在最终选择的特征上进行样本降维可视化...")
if not X_train_final.empty:
    analysis_features(pd.concat([X_train_final, y_train], axis=1), y_train[labels[0]], methods=None)
#特征权重
if feat_coef:
    feat_coef_df = pd.DataFrame(feat_coef, columns=['feature_name', 'Coefficients'])
    # 仅筛选非零系数的特征进行绘图
    feat_coef_df = feat_coef_df[abs(feat_coef_df['Coefficients']) > COEF_THRESHOLD]
    feat_coef_df = feat_coef_df.sort_values(by='Coefficients')

    feat_coef_df['feature_name'] = feat_coef_df['feature_name'].apply(feature_name)

    ax=feat_coef_df.plot(x='feature_name', y='Coefficients', kind='barh',figsize=(12, 8))
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    ax.set_xlim(-0.11, 0.11)
    plt.tight_layout()
    plt.savefig('img/Rad_feature_weights.png', bbox_inches='tight')
    plt.show()

# 计算并显示效应量 (在训练集上)
print("\n在训练集上为LASSO筛选出的特征计算效应量...")
effect_sizes = []
final_features_from_lasso = lasso_final_features_list[0] if lasso_final_features_list else []

if final_features_from_lasso:
    label_column = labels[0]
    df_for_effect_size = train_data_for_stats[final_features_from_lasso + [label_column]]
    groups = df_for_effect_size[label_column].unique()

    for feature in final_features_from_lasso:
        samples = [df_for_effect_size[feature][df_for_effect_size[label_column] == g] for g in groups]
        samples = [s for s in samples if not s.empty]
        
        if len(samples) > 1:
            f_val, p_val = stats.f_oneway(*samples)
            grand_mean = df_for_effect_size[feature].mean()
            ss_between = sum(len(s) * (s.mean() - grand_mean)**2 for s in samples)
            ss_total = sum((x - grand_mean)**2 for x in df_for_effect_size[feature])
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            effect_sizes.append({'Feature': feature, 'Eta-Squared': eta_sq, 'p-value': p_val})
        else:
            effect_sizes.append({'Feature': feature, 'Eta-Squared': 'N/A', 'p-value': 'N/A'})

    if effect_sizes:
        effect_sizes_df = pd.DataFrame(effect_sizes).sort_values(by='Eta-Squared', ascending=False)
        print("\nLASSO筛选出特征的效应量:")
        print(effect_sizes_df)
        effect_sizes_df.to_csv('results/lasso_features_effect_sizes.csv', index=False)

# 6. 保存最终处理好的数据集
print("\n步骤 6: 保存最终处理好的训练集和测试集...")
# 保存经过mRMR筛选后的特征数据
X_train_final.to_csv('results/X_train_final.csv', index=False)
X_test_final.to_csv('results/X_test_final.csv', index=False)

# 保存对应的标签数据
y_train.to_csv('results/y_train.csv', index=False)
y_test.to_csv('results/y_test.csv', index=False)
