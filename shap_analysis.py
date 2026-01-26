import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
modals = [
    "T1-Caudate Nucleus", "T1-Globus Pallidus", "T1-Ventral Midbrain",
    "T1-Nucleus accumbens", "T1-Putamen", "T1-Thalamus",
    "T2-Caudate Nucleus", "T2-Globus Pallidus", "T2-Ventral Midbrain",
    "T2-Nucleus accumbens", "T2-Putamen", "T2-Thalamus"
]

X_train_raw = pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/X_train_final.csv')
y_train = pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/y_train.csv').values.ravel()
X_test = pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/X_test_final.csv')
X_test_raw=pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/X_test_final.csv')
y_test = pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/y_test.csv').values.ravel()
# 打印原始列名，让“隐形”问题显形
print(f"原始 X_train 的列 ({len(X_train_raw.columns)}): {X_train_raw.columns.tolist()}")
print(f"原始 X_test 的列 ({len(X_test_raw.columns)}): {X_test_raw.columns.tolist()}")


# --- 步骤 2: 统一和对齐特征列 ---
print("\n--- 步骤 2: 统一并对齐特征列 ---")
# 移除最常见的“元凶”：'Unnamed: 0' 索引列
if 'Unnamed: 0' in X_train_raw.columns:
    X_train = X_train_raw.drop(columns=['Unnamed: 0'])
    print("从 X_train 中移除了 'Unnamed: 0' 列。")
else:
    X_train = X_train_raw

if 'Unnamed: 0' in X_test_raw.columns:
    X_test = X_test_raw.drop(columns=['Unnamed: 0'])
    print("从 X_test 中移除了 'Unnamed: 0' 列。")
else:
    X_test = X_test_raw

# 关键一步：以训练集为“唯一标准”，确保测试集特征完全一致
final_feature_names = X_train.columns
X_test = X_test[final_feature_names]

print(f"\n对齐后，用于训练和分析的特征 ({len(final_feature_names)}): {final_feature_names.tolist()}")
print(f"对齐后 X_train 的形状: {X_train.shape}")
print(f"对齐后 X_test 的形状: {X_test.shape}")
#训练模型
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42, 
                          learning_rate=0.2,max_depth=12,min_child_weight=10,subsample=0.8,colsample_bytree=1,
                          colsample_bylevel=1,gamma=0.5 ,n_estimators=1000,reg_alpha=1,reg_lambda=1.5,eval_metric='mlogloss'
,use_label_encoder=False)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("SHAP值计算完成。")

print("\n--- SHAP值结构调试信息 ---")
print(f"type(shap_values): {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"len(shap_values) (number of classes): {len(shap_values)}")
    for k, sv in enumerate(shap_values):
        print(f"shap_values[{k}].shape: {sv.shape}")
else:
    print(f"shap_values.shape: {shap_values.shape}") # 如果是三维数组，会打印 (68, 14, 3)
print(f"X_test.shape (after shap_values calculation): {X_test.shape}")
print(f"len(final_feature_names): {len(final_feature_names)}")
print("--------------------------")

img_dir = 'C:/Users/0304/Desktop/onekey-main/img'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
# --- 自动获取特征和类别名称 ---
feature_names = X_test.columns

# 定义一个函数来简化特征名称
def simplify_feature_name(name):
    # 示例简化规则
   name = name.replace('original_glcm_Correlation_T1-Thalamus', 'Thal GLCM Correlation')
   name = name.replace('original_firstorder_Skewness_T1-Nucleus accumbens', 'NAcc Skewness')
   name = name.replace('original_glcm_Imc1_T1-Putamen', 'Put Imc1')
   name = name.replace('original_gldm_SmallDependenceLowGrayLevelEmphasis_T1-Putamen', 'Put SmallDependence')
   name = name.replace('original_glrlm_GrayLevelVariance_T2-Ventral Midbrain', 'VM GrayLevelVariance')
   name = name.replace('original_shape_Maximum3DDiameter_T2-Putamen', 'Put Max3DDiameter')
   name = name.replace('original_shape_Flatness_T2-Putamen', 'Put Flatness')
   name = name.replace('original_shape_VoxelVolume_T2-Nucleus accumbens', 'NAcc VoxelVolume')
   name=name.replace('original_gldm_SmallDependenceLowGrayLevelEmphasis_T2-Nucleus accumbens','NAcc SmallDependence')
   name=name.replace('original_shape_Sphericity_T2-Caudate Nucleus','Cau Sphericity')
   name=name.replace('original_shape_Flatness_T2-Thalamus','Thal Flatness')
   name=name.replace('original_glszm_SmallAreaEmphasis_T1-Putamen','Put SmallAreaEmphasis')
   name=name.replace('original_firstorder_Skewness_T1-Putamen','Put Skewness')
   name=name.replace('original_shape_SurfaceVolumeRatio_T2-Caudate Nucleus','Cau SurfaceVolumeRatio')
   name=name.replace('original_firstorder_Variance_T2-Ventral Midbrain','VM Variance')
   name=name.replace('original_shape_SurfaceVolumeRatio_T2-Nucleus accumbens','NAcc SurfaceVolumeRatio')
   name=name.replace('original_firstorder_Kurtosis_T1-Nucleus accumbens','NAcc Kurtosis')
   name=name.replace('original_glcm_ClusterShade_T1-Nucleus accumbens','NAcc ClusterShade')
    # 可以根据需要添加更多规则
   return name

# 应用简化函数到所有特征名称
simplified_feature_names = [simplify_feature_name(name) for name in feature_names]
# 为图例和标题定义更友好的类别名称
class_display_names = ['SWEDD', 'PD', 'HC']
# 从 y_test 自动提取唯一的类别标签，并创建名称列表
unique_classes = np.unique(y_test)
class_names = [f'Class_{c}' for c in unique_classes]

# 生成并保存一个综合的summary bar plot
plt.figure(figsize=(8, 6)) # 增加图表宽度以适应简化后的名称
plt.yticks(weight='bold',fontsize=18) # Y轴字体加粗
plt.xticks([0, 0.25, 0.5, 0.75, 1.0], weight='bold',fontsize=20)
shap.summary_plot(shap_values, X_test, feature_names=simplified_feature_names, plot_type="bar", class_names=class_display_names, show=False)
plt.title('Mean SHAP Value Magnitude (Overall Feature Importance)',fontweight='bold',fontsize=18)
plt.tight_layout() # 自动调整布局以防止标签重叠
save_path_bar = os.path.join(img_dir, 'SHAP_summary_bar.png')
plt.savefig(save_path_bar, bbox_inches='tight', dpi=300)
plt.show()
print(f"已保存 SHAP summary bar plot 到: {save_path_bar}")

for i, class_name in enumerate(class_names):
    fig, ax = plt.subplots(figsize=(25, 8)) # 增加图表宽度
    print(f"--- 调试信息 for {class_name} ---")
    
    # 修正后的 SHAP 值切片方式
    if isinstance(shap_values, list):
        current_shap_values = shap_values[i]
    else: # 假设是 (samples, features, classes) 的三维数组
        current_shap_values = shap_values[:, :, i] # 取出所有样本、所有特征，针对第 i 个类别
    print(f"shap_values[{i}].shape: {shap_values[i].shape}") # 打印当前类别的SHAP值形状
    print(f"X_test.shape: {X_test.shape}") # 打印X_test的形状
    print(f"len(final_feature_names): {len(final_feature_names)}") # 打印特征数量
    # 使用列表索引 i，它直接对应于 shap_values 列表中的类别

    shap.summary_plot(current_shap_values, X_test, feature_names=simplified_feature_names, show=False)
    ax = plt.gca() # 获取当前的Axes对象
    # 设置 Y 轴字体样式
    ax.set_yticks(ax.get_yticks()) # 保持现有刻度
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold', fontsize=12)
    # 设置 X 轴刻度值和字体样式
    x_ticks_values = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    x_tick_labels = ['-1.0', '', '-0.5', '', '0.0', '', '0.5', '', '1.0']
    ax.set_xticks(x_ticks_values)
    ax.set_xticklabels(x_tick_labels, weight='bold', fontsize=18)
    
    ax.set_title(f'SHAP Beeswarm Plot for {class_name}', fontweight='bold', fontsize=18)
    
    plt.tight_layout() # 自动调整布局以防止标签重叠
    ax.set_xlim(min(x_ticks_values) - 0.2, max(x_ticks_values) + 0.2) # 调整x轴范围
    fig.canvas.draw() # 强制重绘
    
    save_path = os.path.join(img_dir, f'SHAP_beeswarm_{class_name.replace(" ", "_")}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig) # 关闭当前图形
    print(f"已保存 {class_name} 的SHAP beeswarm plot 到: {save_path}")

   

# --- 绘制特定特征的依赖图 for Class 1 (PD) ---
print("\n--- 正在为 Class 1 (PD) 生成依赖图 ---")

# 1. 定义要绘制的 3 个核心特征（确保使用原始特征名）
features_to_plot = [
    'original_firstorder_Skewness_T1-Putamen',
    'original_shape_Maximum3DDiameter_T2-Putamen'
    
]
shap_values_reshaped = np.transpose(shap_values, (2, 0, 1))
# 2. 确定你要分析的类别（例如 PD 对应的索引是 1）
class_index = 1 
class_display_name = class_display_names[class_index]

# 3. 【核心修正】提取对应类别的 SHAP 矩阵并对齐 X_test
# 既然你的 shap_values 只包含这 3 个特征，必须同步过滤 X_test
class_shap_values = shap_values_reshaped[class_index] 
# 打印确认：这里必须显示 (68, 14)
print(f"SHAP 矩阵形状: {class_shap_values.shape}") 
# 打印确认：这里也必须显示 (68, 14)
print(f"X_test 矩阵形状: {X_test.shape}")

for feature in features_to_plot:
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 调用绘制函数
    shap.dependence_plot(
        feature,
        class_shap_values, # 维度：(n, 3)
        X_test,        # 维度：(n, 3) - 必须对齐！
        ax=ax,
        show=False,
        interaction_index="auto" 
    )

    # 4. 美化：简化标题和标签名（参考你之前的简化函数）
    simplified_feature = simplify_feature_name(feature)
    ax.set_title(f'SHAP Dependence Plot for {simplified_feature}\n(Target: {class_display_name})', 
                 fontweight='bold', fontsize=16)
    ax.set_ylabel(f'SHAP Value for {simplified_feature}', fontsize=18, fontweight='bold')
    
    # 美化 X 轴
    xlabel = ax.get_xlabel()
    ax.set_xlabel(simplify_feature_name(xlabel), fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    # 5. 处理 Colorbar 标签简化
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[1]
        raw_interaction_name = cbar_ax.get_ylabel()
        cbar_ax.set_ylabel(simplify_feature_name(raw_interaction_name), 
                           fontsize=18, fontweight='bold')

    plt.tight_layout()
    
    # 6. 自动保存
    save_name = f'SHAP_Dep_{simplified_feature.replace(" ", "_")}_{class_display_name}.png'
    save_path = os.path.join(img_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)
    print(f"已保存: {save_path}")

# --- 绘制特定特征的依赖图 for Class 0 (SWEDD) ---
print("\n--- 正在为 Class 0 (SWEDD) 生成依赖图 ---")

# 1. 定义要绘制的 3 个核心特征（确保使用原始特征名）
features_to_plot_0 = [
    'original_shape_Maximum3DDiameter_T2-Putamen'

    
]
shap_values_reshaped = np.transpose(shap_values, (2, 0, 1))
# 2. 确定你要分析的类别（例如 PD 对应的索引是 1）
class_index_0 = 0 
class_display_name_0 = class_display_names[class_index_0]

# 3. 【核心修正】提取对应类别的 SHAP 矩阵并对齐 X_test
# 既然你的 shap_values 只包含这 3 个特征，必须同步过滤 X_test
class_shap_values = shap_values_reshaped[class_index_0]
# 打印确认：这里必须显示 (68, 14)
print(f"SHAP 矩阵形状: {class_shap_values.shape}") 
# 打印确认：这里也必须显示 (68, 14)
print(f"X_test 矩阵形状: {X_test.shape}")

for feature in features_to_plot_0:
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 调用绘制函数
    shap.dependence_plot(
        feature,
        class_shap_values, # 维度：(n, 3)
        X_test,        # 维度：(n, 3) - 必须对齐！
        ax=ax,
        show=False,
        interaction_index="auto" 
    )

    # 4. 美化：简化标题和标签名（参考你之前的简化函数）
    simplified_feature = simplify_feature_name(feature)
    ax.set_title(f'SHAP Dependence Plot for {simplified_feature}\n(Target: {class_display_name_0})', 
                 fontweight='bold', fontsize=16)
    ax.set_ylabel(f'SHAP Value for {simplified_feature}', fontsize=14, fontweight='bold')
    
    # 美化 X 轴
    xlabel = ax.get_xlabel()
    ax.set_xlabel(simplify_feature_name(xlabel), fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    # 5. 处理 Colorbar 标签简化
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[1]
        raw_interaction_name = cbar_ax.get_ylabel()
        cbar_ax.set_ylabel(simplify_feature_name(raw_interaction_name), 
                           fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    # 6. 自动保存
    save_name = f'SHAP_Dep_{simplified_feature.replace(" ", "_")}_{class_display_name}.png'
    save_path = os.path.join(img_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)
    print(f"已保存: {save_path}")