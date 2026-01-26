import os
import nibabel as nib
from pathlib import Path
import pandas as pd
#留一法
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
import onekey_algo.custom.components as okcomp
from onekey_algo.custom.components.delong import calc_95_CI
from scipy.special import expit
from onekey_algo.custom.components.comp1 import draw_roc
# 导入 LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.utils import resample
#加载数据
X_train=pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/X_train_final.csv')
y_train=pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/y_train.csv').values.ravel()
X_test=pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/X_test_final.csv')
y_test=pd.read_csv('C:/Users/0304/Desktop/onekey-main/results/y_test.csv').values.ravel()
def dca_curves(y_true, y_probas, thresholds=None,
               mode="snb",          # "raw" | "snb" | "minmax" | "minmax_robust"
               robust_q=(5,95),     # 用于 minmax_robust
               eps=1e-6):
     """
     mode:
      - "raw": 原始 NB（推荐）
      - "snb": 标准化净获益 NB / prevalence（强烈推荐做“归一化”时用）
      - "minmax": 三条曲线联合 min–max 到 [0,1]
      - "minmax_robust": 用 5–95 分位联合缩放再 clip 到 [0,1]
     """
     print(f"DEBUG: dca_curves - y_true shape: {np.asarray(y_true).shape}, y_probas shape: {np.asarray(y_probas).shape}") 
     """
    为多分类问题计算决策曲线分析 (DCA)。
    采用“一对多”策略，为每个类别计算净获益。
    返回一个包含每个类别结果的字典。
    """
     results = {}
     y_true_df = pd.DataFrame(y_true)
     classes = sorted(y_true_df.iloc[:, 0].unique())
     y_true_binned = label_binarize(y_true, classes=classes)

     if y_probas.shape[1] != len(classes):
        raise ValueError("预测概率的列数与真实标签的类别数不匹配。")

     if thresholds is None:
        thresholds = np.linspace(0.01, 0.80, 80)

     for i, class_label in enumerate(classes):
        y = y_true_binned[:, i]
        p = y_probas[:, i]
        N = len(y)
        prev = y.mean()

        nb_model = []
        for pt in thresholds:
            odds = pt / max(1 - pt, eps)
            pred = (p >= pt).astype(int)
            TP = np.sum((pred == 1) & (y == 1))
            FP = np.sum((pred == 1) & (y == 0))
            nb = (TP / N) - (FP / N) * odds
            nb_model.append(nb)
        
        nb_model = np.array(nb_model)
        odds_all = thresholds / np.maximum(1 - thresholds, eps)
        nb_all = prev - (1 - prev) * odds_all
        nb_none = np.zeros_like(thresholds)

        if mode == "snb":
            scale = max(prev, eps)
            nb_model /= scale
            nb_all /= scale
        elif mode in ("minmax", "minmax_robust"):
            all_vals = np.concatenate([nb_model, nb_all, nb_none])
            if mode == "minmax":
                lo, hi = np.min(all_vals), np.max(all_vals)
            else:
                lo, hi = np.percentile(all_vals, robust_q)
            if hi - lo < 1e-12:
                hi = lo + 1e-12
            nb_model = np.clip((nb_model - lo) / (hi - lo), 0, 1)
            nb_all = np.clip((nb_all - lo) / (hi - lo), 0, 1)

        results[class_label] = (thresholds, nb_model, nb_all, nb_none)
        
     return results
     

def plot_multiclass_dca(y_true, y_probas, model_name, savepath=None):
    """
    绘制多分类决策曲线分析图。
    """
    plt.figure(figsize=(7, 5))
    
    # 一次性计算所有类别的DCA曲线
    dca_results = dca_curves(y_true, y_probas, mode="snb")
    class_mapping = {0: 'SWEDD', 1: 'PD', 2: 'HC'}
    # 为每个类别绘制模型曲线
    for class_label, (thr, nb_m, _, _) in dca_results.items():
        display_label = class_mapping.get(class_label, f'Class {class_label}')
        plt.plot(thr, nb_m, lw=4, label=f'{model_name} ({display_label})')

    # 使用第一个类别的结果来绘制 "Treat all" 和 "Treat none" 作为参考
    first_class_label = sorted(dca_results.keys())[0]
    thr_example, _, nb_all_example, nb_none_example = dca_results[first_class_label]
    
    plt.plot(thr_example, nb_all_example, '-', label='Treat all(ref)', color='black', lw=4)
    plt.plot(thr_example, nb_none_example, '--', label='Treat none', color='black', lw=4)
    
    plt.ylim(-0.1, 1) # 调整Y轴范围以更好地显示
    plt.xlabel('Threshold probability', fontweight='bold',fontsize=15)
    plt.ylabel('Net benefit (standardized)', fontweight='bold',fontsize=15)
    plt.grid(True, alpha=0.4)
    font = FontProperties(size=12,weight='bold')
    plt.legend(loc='upper right', prop=font)
    plt.title(f'Decision Curve Analysis for {model_name}', fontsize=15,fontweight='bold')
    plt.tight_layout()
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
        label.set_fontsize(12)
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

def bootstrap_auc_ci(y_true, y_proba, n_bootstraps=1000, alpha=0.05, average='macro'):
    """Calculates bootstrap CI for multiclass AUC."""
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    if n_classes < 2:
        return (np.nan, np.nan)
        
    y_true_bin = label_binarize(y_true, classes=classes)
    
    bootstrapped_scores = []
    # use a fixed random state for reproducibility
    rng = np.random.RandomState(42) 

    for i in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        
        # if bootstrap sample does not have all classes, skip it
        if len(np.unique(y_true[indices])) < n_classes:
            continue
            
        try:
            score = roc_auc_score(y_true_bin[indices], y_proba[indices], multi_class='ovr', average=average)
            bootstrapped_scores.append(score)
        except ValueError:
            # This can happen if a class has only one label in the resampled data.
            continue

    if not bootstrapped_scores:
        return (np.nan, np.nan)

    lower_p = 100 * (alpha / 2.0)
    upper_p = 100 * (1 - alpha / 2.0)
    lower = np.percentile(bootstrapped_scores, lower_p)
    upper = np.percentile(bootstrapped_scores, upper_p)
    
    return (lower, upper)

def bootstrap_auc_ci_per_class(y_true_class, y_proba_class, n_bootstraps=1000, alpha=0.05):
    """Calculates bootstrap CI for a single class's AUC."""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        indices = rng.choice(len(y_true_class), len(y_true_class), replace=True)
        
        # if bootstrap sample does not have both classes, skip it
        if len(np.unique(y_true_class[indices])) < 2:
            continue
            
        try:
            score = roc_auc_score(y_true_class[indices], y_proba_class[indices])
            bootstrapped_scores.append(score)
        except ValueError:
            # This can happen if a class has only one label in the resampled data.
            continue

    if not bootstrapped_scores:
        return (np.nan, np.nan)
    lower_p = 100 * (alpha / 2.0)
    upper_p = 100 * (1 - alpha / 2.0)
    lower = np.percentile(bootstrapped_scores, lower_p)
    upper = np.percentile(bootstrapped_scores, upper_p)
    
    return (lower, upper)

def compute_metrics(y_true_all, y_proba_all):
    classes = sorted(np.unique(y_true_all))
    n_classes = len(classes)
    y_pred_all = np.argmax(y_proba_all, axis=1)

    metrics = {}

    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)

    # --- Accuracy ---
    # Overall accuracy (is the same as micro-averaged accuracy)
    overall_accuracy = accuracy_score(y_true_all, y_pred_all)
    metrics['accuracy'] = overall_accuracy
    metrics['accuracy_micro'] = overall_accuracy

    # Per-class, macro, and weighted accuracy
    per_class_accuracy = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        class_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        per_class_accuracy.append(class_acc)
        metrics[f'accuracy_class_{classes[i]}'] = class_acc
    
    metrics['accuracy_macro'] = np.mean(per_class_accuracy)
    if n_classes > 0:
        support = cm.sum(axis=1)
        if support.sum() > 0:
            metrics['accuracy_weighted'] = np.average(per_class_accuracy, weights=support)
        else:
            metrics['accuracy_weighted'] = 0.0
    else:
        metrics['accuracy_weighted'] = 0.0

    # --- Precision, Recall (Sensitivity), F1-score ---
    precision_per_class = precision_score(y_true_all, y_pred_all, labels=classes, average=None, zero_division=0)
    recall_per_class = recall_score(y_true_all, y_pred_all, labels=classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_all, y_pred_all, labels=classes, average=None, zero_division=0)
    
    for i, cls in enumerate(classes):
        metrics[f'precision_class_{cls}'] = precision_per_class[i]
        metrics[f'sensitivity_class_{cls}'] = recall_per_class[i]
        metrics[f'f1_class_{cls}'] = f1_per_class[i]

    for avg in ['micro', 'macro', 'weighted']:
        metrics[f'precision_{avg}'] = precision_score(y_true_all, y_pred_all, average=avg, zero_division=0)
        metrics[f'sensitivity_{avg}'] = recall_score(y_true_all, y_pred_all, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(y_true_all, y_pred_all, average=avg, zero_division=0)

    # --- Specificity ---
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    specificity_per_class = []
    total_fp_sum = 0
    total_tn_sum = 0
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(specificity)
        metrics[f'specificity_class_{classes[i]}'] = specificity
        total_fp_sum += fp
        total_tn_sum += tn

    if (total_tn_sum + total_fp_sum) > 0:
        metrics['specificity_micro'] = total_tn_sum / (total_tn_sum + total_fp_sum)
    else:
        metrics['specificity_micro'] = 0.0
    
    metrics['specificity_macro'] = np.mean(specificity_per_class)
    if n_classes > 0:
        support = cm.sum(axis=1)
        if support.sum() > 0:
            metrics['specificity_weighted'] = np.average(specificity_per_class, weights=support)
        else:
            metrics['specificity_weighted'] = 0.0
    else:
        metrics['specificity_weighted'] = 0.0


      # --- AUC ---
    if n_classes > 2:
        y_true_bin = label_binarize(y_true_all, classes=classes)
        for i, cls in enumerate(classes):
            # Handle case where a class has only one sample in y_true_bin
            if len(np.unique(y_true_bin[:, i])) > 1:
                auc_val = roc_auc_score(y_true_bin[:, i], y_proba_all[:, i])
                metrics[f'auc_class_{cls}'] = auc_val
                try:
                    auc_ci = bootstrap_auc_ci_per_class(y_true_bin[:, i], y_proba_all[:, i])
                    metrics[f'95%CI_class_{cls}'] = f"{auc_ci[0]:.3f}-{auc_ci[1]:.3f}"
                except Exception:
                    metrics[f'95%CI_class_{cls}'] = "N/A"
            else:
                metrics[f'auc_class_{cls}'] = np.nan
                metrics[f'95%CI_class_{cls}'] = "N/A"
        
        # Micro-average AUC
        try:
            micro_auc = roc_auc_score(y_true_bin, y_proba_all, average="micro")
            metrics['auc_micro'] = micro_auc
            try:
                auc_ci = bootstrap_auc_ci(y_true_all, y_proba_all, average='micro')
                metrics['95%CI_micro'] = f"{auc_ci[0]:.3f}-{auc_ci[1]:.3f}"
            except Exception:
                metrics['95%CI_micro'] = "N/A"
        except ValueError:
            metrics['auc_micro'] = np.nan
            metrics['95%CI_micro'] = "N/A"
        
        # Macro and Weighted AUC
        for avg in ['macro', 'weighted']:
            try:
                avg_auc = roc_auc_score(y_true_bin, y_proba_all, multi_class='ovr', average=avg)
                metrics[f'auc_{avg}'] = avg_auc
                if avg == 'macro':
                    try:
                        # Using bootstrapping for Macro AUC CI
                        auc_ci = bootstrap_auc_ci(y_true_all, y_proba_all, average='macro')
                        metrics['95%CI_macro'] = f"{auc_ci[0]:.3f}-{auc_ci[1]:.3f}"
                    except Exception:
                        metrics['95%CI_macro'] = "N/A"
                else:
                    metrics[f'95%CI_{avg}'] = "N/A"  # CI for weighted is also complex
            except ValueError:
                metrics[f'auc_{avg}'] = np.nan
                metrics[f'95%CI_{avg}'] = "N/A"

    elif n_classes == 2: # binary
        # Ensure y_true_all is binary 0/1 for roc_auc_score
        y_true_binary = (y_true_all == classes[1]).astype(int)
        auc_score_val = roc_auc_score(y_true_binary, y_proba_all[:, 1])
        metrics['auc_class_0'] = np.nan # Not directly computed for class 0
        metrics['auc_class_1'] = auc_score_val
        metrics['auc_micro'] = auc_score_val
        metrics['auc_macro'] = auc_score_val
        metrics['auc_weighted'] = auc_score_val
        try:
            auc_ci = bootstrap_auc_ci_per_class(y_true_binary, y_proba_all[:, 1])
            ci_str = f"{auc_ci[0]:.3f}-{auc_ci[1]:.3f}"
            metrics['95%CI_class_1'] = ci_str
            metrics['95%CI_micro'] = ci_str
            metrics['95%CI_macro'] = ci_str
            metrics['95%CI_weighted'] = ci_str
        except Exception:
            metrics['95%CI_class_1'] = "N/A"
            metrics['95%CI_micro'] = "N/A"
            metrics['95%CI_macro'] = "N/A"
            metrics['95%CI_weighted'] = "N/A"

    return metrics
def draw_roc(y_trues, y_probas, labels=None, title='ROC', ax=None,plot_type='all'):
    """
    绘制ROC曲线，支持多分类
    :param y_trues: list of y_true
    :param y_probas: list of y_proba
    :param labels: list of labels
    :param title: 标题
    :param ax: matplotlib axes
    """
    if not isinstance(y_trues, list):
        y_trues = [y_trues]
        y_probas = [y_probas]
    if not labels:
        labels = [f'model_{i}' for i in range(len(y_trues))]

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for y_true, y_proba, label in zip(y_trues, y_probas, labels):
        y_true = np.asarray(y_true).ravel()
        y_proba = np.asarray(y_proba)
        classes = np.unique(y_true)
        n_classes = len(classes)

        if n_classes > 2:
            # Multi-class case
            y_true_bin = label_binarize(y_true, classes=classes)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute macro-average ROC curve and ROC area
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            if plot_type in ['all', 'averages']:
                ax.plot(fpr["micro"], tpr["micro"],
                        label=f'{label} micro-average ROC (area = {roc_auc["micro"]:0.3f})',
                        color='deeppink', linestyle=':', linewidth=4)

                ax.plot(fpr["macro"], tpr["macro"],
                        label=f'{label} macro-average ROC (area = {roc_auc["macro"]:0.3f})',
                        color='navy', linestyle=':', linewidth=4)

            if plot_type in ['all', 'classes']:
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
                class_labels = {0: 'SWEDD', 1: 'PD', 2: 'HC'}
                for i, color in zip(range(n_classes), colors):
                    class_name = class_labels.get(classes[i], f'Class {classes[i]}')  
                    ax.plot(fpr[i], tpr[i], color=color, lw=4,
                            label=f'{label} ROC curve of {class_name}(area = {roc_auc[i]:0.3f})')#右下角标注设置
        else:
            # Binary case
            fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=4, label=f'{label} (AUC = %0.3f)' % roc_auc)

    ax.plot([0, 1], [0, 1], 'k--', lw=4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontweight='bold')
    ax.set_ylabel('True Positive Rate',fontweight='bold')
    ax.set_title(title,fontweight='bold')
    font = FontProperties(size=12,weight='bold')
    ax.legend(loc="lower right",prop=font)

# 用于存储每个模型的总体指标
final_metrics_list = []
train_results={}
test_results={}

#1.LR
LR_metrics_list = []

# 留一法训练
#param_grid_LR={'C': [0.01, 0.1, 1, 10], 'penalty': ['l2','l1', 'elasticnet'],'max_iter': [100,500,800,1000,2000,5000,10000],'solver': ['liblinear', 'saga','newton-cg','lbfgs'],'tol': [1e-6,1e-5,1e-4, 1e-3,1e-2],'fit_intercept': [True, False],'dual': [True, False],'intercept_scaling': [1, 10, 100]}
param_grid_LR = [
    {
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'penalty': [None],
        'max_iter': [500,1000, 2000, 5000],
        'tol': [1e-5, 1e-4, 1e-3]
    }
]
grid_search_LR = GridSearchCV(LogisticRegression('multinomial',class_weight='balanced', random_state=42), param_grid_LR, cv=10, scoring='roc_auc_ovr', n_jobs=-1)
grid_search_LR.fit(X_train, y_train)
print(grid_search_LR.best_params_)
print(f"Best score: {grid_search_LR.best_score_}")
# 获取最优模型
#best_model_LR = LogisticRegression(class_weight='balanced', random_state=42,C=0.01,penalty='l2',solver='liblinear',max_iter=100)
#best_model_LR.fit(X_train, y_train)
best_model_LR = grid_search_LR.best_estimator_
    
    # 训练集预测
if hasattr(best_model_LR,"predict_proba"):
    proba_train_LR = best_model_LR.predict_proba(X_train)
elif hasattr(best_model_LR,"decision_function"):
    proba_train_LR=expit(best_model_LR.decision_function(X_train))
    
pred_train_LR = np.argmax(proba_train_LR, axis=1)
pred_train_LR=np.atleast_1d(pred_train_LR)
    

    # 测试集预测
if hasattr(best_model_LR,"predict_proba"):
    proba_test_LR = best_model_LR.predict_proba(X_test)
elif hasattr(best_model_LR,"decision_function"):
    proba_test_LR=expit(best_model_LR.decision_function(X_test))

pred_test_LR = np.argmax(proba_test_LR, axis=1)
pred_test_LR=np.atleast_1d(pred_test_LR)


# 计算LogisticRegression的评估指标
train_metrics_LR = compute_metrics(y_train, proba_train_LR)
test_metrics_LR = compute_metrics(y_test, proba_test_LR)

train_results['LR'] = (y_train, proba_train_LR, pred_train_LR)  
test_results['LR'] = (y_test, proba_test_LR, pred_test_LR) 

all_metrics = {'train': train_metrics_LR, 'test': test_metrics_LR} 
classes = sorted(np.unique(y_train))
for dataset_name, metrics in all_metrics.items():
    # Per-class rows
    for cls in classes:
        row = {
            'model': 'LR',
            'dataset': dataset_name,
            'class/avg': f'class_{cls}',
            'accuracy': metrics.get(f'accuracy_class_{cls}'), # Overall accuracy, repeated for context
            'auc': metrics.get(f'auc_class_{cls}'),
            'sensitivity': metrics.get(f'sensitivity_class_{cls}'),
            'specificity': metrics.get(f'specificity_class_{cls}'),
            'precision': metrics.get(f'precision_class_{cls}'),
            'f1': metrics.get(f'f1_class_{cls}'),
            '95%CI': metrics.get('95%CI')
        }
        LR_metrics_list.append(row)

    # Averaged rows
    for avg in ['micro', 'macro', 'weighted']:
        row = {
            'model': 'LR',
            'dataset': dataset_name,
            'class/avg': avg,
            'accuracy': metrics.get(f'accuracy_{avg}'), # Accuracy is overall, often reported with micro avg
            'auc': metrics.get(f'auc_{avg}'),
            'sensitivity': metrics.get(f'sensitivity_{avg}'),
            'specificity': metrics.get(f'specificity_{avg}'),
            'precision': metrics.get(f'precision_{avg}'),
            'f1': metrics.get(f'f1_{avg}'),
            '95%CI': metrics.get('95%CI')
        }
        LR_metrics_list.append(row)
# 将 final_metrics_list 转换为 DataFrame
LR_metrics_df = pd.DataFrame(LR_metrics_list)

# 保存为 CSV 文件
LR_metrics_df.to_csv('results/LR_model_metrics.csv', index=False)

# 绘制LogisticRegression的ROC曲线
plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_LR, proba_test_LR], 
         labels=['Train', 'Test'], title=f'Model: LR (Macro/Micro AVG)',
         plot_type='averages')
plt.savefig(f'img/Rad_model_LR_roc_averages.png', bbox_inches='tight')
plt.show()

# 绘制 per-class ROC 曲线
plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_LR, proba_test_LR], 
         labels=['Train', 'Test'], title=f'Model: LR (Per-Class)',
         plot_type='classes')
plt.savefig(f'img/Rad_model_LR_roc_classes.png', bbox_inches='tight')
plt.show()
#绘制混淆矩阵
cm_test = confusion_matrix(y_test, pred_test_LR)
labels = sorted(np.unique(y_test))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,annot_kws={"size": 40,"weight": "bold"}
            ,vmin=0,vmax=20,cbar_kws={'ticks': [0, 5, 10, 15, 20]})
plt.title(f'Confusion Matrix for LR',fontweight='bold',fontsize=30)
plt.ylabel('True label',fontweight='bold',fontsize=30)
plt.xlabel('Predicted label',fontweight='bold',fontsize=30)
plt.savefig(f'img/LR_Confusion_Matrix.png')
plt.show()
plt.close()
#绘制DCA决策曲线
plot_multiclass_dca(y_test, proba_test_LR, model_name='LR', savepath="img/LR_DCA.png")


#2.SVM模型
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
#特征数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

SVM_metrics_list = []
# 创建模型字典
#model_names = ['SVM']  # 这里选择您想使用的模型名称（如 LR）
#models_dict = okcomp.comp1.create_clf_model(model_names)
#获取模型
SVM_model = SVC(probability=True, random_state=42)

# 留一法训练
search_spaces_SVM=[{'kernel': ['linear'], 'C': [0.01,0.1, 1, 10,100], 'max_iter': [1000, 5000, -1], 'tol': [1e-6,1e-5,1e-4, 1e-3,1e-2,0.1],'cache_size': [200, 500, 1000],'shrinking': [True, False]},
    {'kernel': ['rbf'], 'C': [0.01,0.1, 1, 10,100], 'gamma': [0.01, 0.1, 'scale'], 'max_iter': [200,500,800,1000, 5000, -1], 'tol': [1e-6,1e-5,1e-4, 1e-3,1e-2,0.1],'cache_size': [200, 500, 1000],'shrinking': [True, False]}
]

bayes_search_SVM = BayesSearchCV(
    estimator=SVM_model,
    search_spaces=search_spaces_SVM,
    n_iter=50,
    cv=10,
    scoring='roc_auc_ovr',
    n_jobs=-1,
    random_state=42
)
bayes_search_SVM.fit(X_train_sc, y_train)
print(bayes_search_SVM.best_params_)
print(f"Best score: {bayes_search_SVM.best_score_}")
# 获取最优模型
#best_model_SVM = SVC(C=0.1, kernel='linear', probability=True, random_state=42,gamma='scale')
#best_model_SVM.fit(X_train, y_train)
best_model_SVM = bayes_search_SVM.best_estimator_
    
    # 训练集预测
if hasattr(best_model_SVM,"predict_proba"):
    proba_train_SVM = best_model_SVM.predict_proba(X_train_sc)
elif hasattr(best_model_SVM,"decision_function"):
    proba_train_SVM=expit(best_model_SVM.decision_function(X_train_sc))
    
pred_train_SVM = np.argmax(proba_train_SVM, axis=1)
pred_train_SVM=np.atleast_1d(pred_train_SVM)
    

    # 测试集预测
if hasattr(best_model_SVM,"predict_proba"):
    proba_test_SVM = best_model_SVM.predict_proba(X_test_sc)
elif hasattr(best_model_SVM,"decision_function"):
    proba_test_SVM=expit(best_model_SVM.decision_function(X_test_sc))

pred_test_SVM = np.argmax(proba_test_SVM, axis=1)
pred_test_SVM=np.atleast_1d(pred_test_SVM)

# 计算SVM的评估指标

train_metrics_SVM = compute_metrics(y_train, proba_train_SVM)
test_metrics_SVM = compute_metrics(y_test, proba_test_SVM)

train_results['SVM'] = (y_train, proba_train_SVM, pred_train_SVM)  
test_results['SVM'] = (y_test, proba_test_SVM, pred_test_SVM) 

all_metrics_SVM = {'train': train_metrics_SVM, 'test': test_metrics_SVM} 
classes = sorted(np.unique(y_train))
SVM_metrics_list=[]
for dataset_name, metrics in all_metrics_SVM.items():
    # Per-class rows
    for cls in classes:
        row = {
            'model': 'SVM',
            'dataset': dataset_name,
            'class/avg': f'class_{cls}',
            'accuracy': metrics.get(f'accuracy_class_{cls}'), # Overall accuracy, repeated for context
            'auc': metrics.get(f'auc_class_{cls}'),
            'sensitivity': metrics.get(f'sensitivity_class_{cls}'),
            'specificity': metrics.get(f'specificity_class_{cls}'),
            'precision': metrics.get(f'precision_class_{cls}'),
            'f1': metrics.get(f'f1_class_{cls}'),
            '95%CI': metrics.get('95%CI')
        }
        SVM_metrics_list.append(row)

    # Averaged rows
    for avg in ['micro', 'macro', 'weighted']:
        row = {
            'model': 'SVM',
            'dataset': dataset_name,
            'class/avg': avg,
            'accuracy': metrics.get(f'accuracy_{avg}'), # Accuracy is overall, often reported with micro avg
            'auc': metrics.get(f'auc_{avg}'),
            'sensitivity': metrics.get(f'sensitivity_{avg}'),
            'specificity': metrics.get(f'specificity_{avg}'),
            'precision': metrics.get(f'precision_{avg}'),
            'f1': metrics.get(f'f1_{avg}'),
            '95%CI': metrics.get('95%CI')
        }
        SVM_metrics_list.append(row)
# 将 final_metrics_list 转换为 DataFrame
SVM_metrics_df = pd.DataFrame(SVM_metrics_list)

# 保存为 CSV 文件
SVM_metrics_df.to_csv('results/SVM_model_metrics.csv', index=False)

# 绘制SVM的ROC曲线
plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_SVM, proba_test_SVM], 
         labels=['Train', 'Test'], title=f'Model: SVM (Macro/Micro AVG)',
         plot_type='averages')
plt.savefig(f'img/Rad_model_SVM_roc_averages.png', bbox_inches='tight')
plt.show()

# 绘制 per-class ROC 曲线
plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_SVM, proba_test_SVM], 
         labels=['Train', 'Test'], title=f'Model: SVM (Per-Class)',
         plot_type='classes')
plt.savefig(f'img/Rad_model_SVM_roc_classes.png', bbox_inches='tight')
plt.show()
#绘制混淆矩阵
cm_test = confusion_matrix(y_test, pred_test_SVM)
labels = sorted(np.unique(y_test))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,annot_kws={"size": 40,"weight": "bold"}
             ,vmin=0,vmax=20,cbar_kws={'ticks': [0, 5, 10, 15, 20]})
plt.title(f'Confusion Matrix for SVM',fontweight='bold',fontsize=30)
plt.ylabel('True label',fontweight='bold',fontsize=30)
plt.xlabel('Predicted label',fontweight='bold',fontsize=30)
plt.savefig(f'img/SVM_Confusion_Matrix.png')
plt.show()
plt.close()
#绘制DCA决策曲线
plot_multiclass_dca(y_test, proba_test_SVM, model_name='SVM', savepath="img/SVM_DCA.png")

#3.XGBoost
from xgboost import XGBClassifier

XGBoost_metrics_list = []
# 创建模型字典
model_names = ['XGBoost']  # 这里选择您想使用的模型名称（如 LR）
models_dict = okcomp.comp1.create_clf_model(model_names)
#获取模型
#XGBoost_model = models_dict['XGBoost'] 
param_grid_XGBoost = {
   'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [5, 10],
            'reg_alpha': [0.1,0.5, 1],
            'reg_lambda': [0.1, 0.5, 1]}
XGBoost_model = XGBClassifier(random_state=42, eval_metric='mlogloss',num_class=3, tree_method='hist',device='cuda')

grid_search_XGBoost = GridSearchCV(XGBoost_model, param_grid_XGBoost, cv=10, scoring='roc_auc_ovr', n_jobs=2)
grid_search_XGBoost.fit(X_train, y_train)
print(f"XGBoost Best params: {grid_search_XGBoost.best_params_}")
print(f"XGBoost Best score: {grid_search_XGBoost.best_score_}")

# 获取最优模型
best_model_XGBoost = grid_search_XGBoost.best_estimator_
# 获取最优模型
#best_model_XGBoost = XGBClassifier(learning_rate=0.01,max_depth=3,min_child_weight=1,n_estimators=50,reg_alpha=0.1,reg_lamba=0.1)
#best_model_XGBoost.fit(X_train, y_train)
   # 训练集预测
if hasattr(best_model_XGBoost,"predict_proba"):
    proba_train_XGBoost = best_model_XGBoost.predict_proba(X_train)
elif hasattr(best_model_XGBoost,"decision_function"):
    proba_train_XGBoost=expit(best_model_XGBoost.decision_function(X_train))
    
pred_train_XGBoost = np.argmax(proba_train_XGBoost, axis=1)
pred_train_XGBoost=np.atleast_1d(pred_train_XGBoost)
    

    # 测试集预测
if hasattr(best_model_XGBoost,"predict_proba"):
    proba_test_XGBoost = best_model_XGBoost.predict_proba(X_test)
elif hasattr(best_model_XGBoost,"decision_function"):
    proba_test_XGBoost=expit(best_model_XGBoost.decision_function(X_test))
    
pred_test_XGBoost = np.argmax(proba_test_XGBoost, axis=1)
pred_test_XGBoost=np.atleast_1d(pred_test_XGBoost)

# 计算XGBoost的评估指标
train_metrics_XGBoost = compute_metrics(y_train, proba_train_XGBoost)
test_metrics_XGBoost = compute_metrics(y_test, proba_test_XGBoost)

train_results['XGBoost'] = (y_train, proba_train_XGBoost, pred_train_XGBoost)  
test_results['XGBoost'] = (y_test, proba_test_XGBoost, pred_test_XGBoost) 

all_metrics_XGBoost = {'train': train_metrics_XGBoost, 'test': test_metrics_XGBoost} 
classes = sorted(np.unique(y_train))
XGBoost_metrics_list=[]
for dataset_name, metrics in all_metrics_XGBoost.items():
    # Per-class rows
    for cls in classes:
        row = {
            'model': 'XGBoost',
            'dataset': dataset_name,
            'class/avg': f'class_{cls}',
            'accuracy': metrics.get(f'accuracy_class_{cls}'), # Overall accuracy, repeated for context
            'auc': metrics.get(f'auc_class_{cls}'),
            'sensitivity': metrics.get(f'sensitivity_class_{cls}'),
            'specificity': metrics.get(f'specificity_class_{cls}'),
            'precision': metrics.get(f'precision_class_{cls}'),
            'f1': metrics.get(f'f1_class_{cls}'),
            '95%CI': metrics.get('95%CI')
        }
        XGBoost_metrics_list.append(row)

    # Averaged rows
    for avg in ['micro', 'macro', 'weighted']:
        row = {
            'model': 'XGBoost',
            'dataset': dataset_name,
            'class/avg': avg,
            'accuracy': metrics.get(f'accuracy_{avg}'), # Accuracy is overall, often reported with micro avg
            'auc': metrics.get(f'auc_{avg}'),
            'sensitivity': metrics.get(f'sensitivity_{avg}'),
            'specificity': metrics.get(f'specificity_{avg}'),
            'precision': metrics.get(f'precision_{avg}'),
            'f1': metrics.get(f'f1_{avg}'),
            '95%CI': metrics.get('95%CI')
        }
        XGBoost_metrics_list.append(row)
# 将 final_metrics_list 转换为 DataFrame
XGB_metrics_df = pd.DataFrame(XGBoost_metrics_list)

# 保存为 CSV 文件
XGB_metrics_df.to_csv('results/XGBoost_model_metrics.csv', index=False)

# 绘制XGBoost的ROC曲线
plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_XGBoost, proba_test_XGBoost], 
         labels=['Train', 'Test'], title=f'Model: XGBoost (Macro/Micro AVG)',
         plot_type='averages')
plt.savefig(f'img/Rad_model_XGBoost_roc_averages.png', bbox_inches='tight')
plt.show()

# 绘制 per-class ROC 曲线
plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_XGBoost, proba_test_XGBoost], 
         labels=['Train', 'Test'], title=f'Model: XGBoost (Per-Class)',
         plot_type='classes')
plt.savefig(f'img/Rad_model_XGBoost_roc_classes.png', bbox_inches='tight')
plt.show()
#绘制混淆矩阵
cm_test = confusion_matrix(y_test, pred_test_XGBoost)
labels = sorted(np.unique(y_test))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,annot_kws={"size": 40,"weight": "bold"}
            ,vmin=0,vmax=20,cbar_kws={'ticks': [0, 5, 10, 15, 20]})
plt.title(f'Confusion Matrix for XGBoost',fontweight='bold',fontsize=30)
plt.ylabel('True label',fontweight='bold',fontsize=30)
plt.xlabel('Predicted label',fontweight='bold',fontsize=30)
plt.savefig(f'img/XGBoost_Confusion_Matrix.png')
plt.show()
plt.close()
#绘制DCA决策曲线
plot_multiclass_dca(y_test, proba_test_XGBoost, model_name='XGBoost', savepath="img/XGBoost_DCA.png")

#4.LightGBM
import lightgbm as lgb  
from lightgbm import LGBMClassifier
LightGBM_metrics_list = []

# 创建模型字典
#model_names = ['LightGBM']  # 这里选择您想使用的模型名称（如 LR）
#models_dict = okcomp.comp1.create_clf_model(model_names)
#获取模型
#LightGBM_model = models_dict['LightGBM'] 
param_grid_LightGBM = {
    'n_estimators': [50,100],'learning_rate': [0.01, 0.1],
    'max_depth': [ 10, 20],'num_leaves': [31, 62],
   'reg_alpha':[0.1, 1],'reg_lambda':[ 0.1,1]}

LightGBM_model = LGBMClassifier(random_state=42)

grid_search_LightGBM = GridSearchCV(LightGBM_model, param_grid_LightGBM, cv=10, scoring='roc_auc_ovr', n_jobs=-1)  
grid_search_LightGBM.fit(X_train, y_train)
print(f"LightGBM Best params: {grid_search_LightGBM.best_params_}")
print(f"LightGBM Best score: {grid_search_LightGBM.best_score_}")   

# 获取最优模型
best_model_LightGBM = grid_search_LightGBM.best_estimator_
# 获取最优模型
#best_model_LightGBM = LGBMClassifier(max_depth=10,learning_rate=0.01,n_estimators=50,reg_alpha=0.1,reg_lambda=0.1,num_leaves=31)

best_model_LightGBM.fit(X_train, y_train)
   # 训练集预测
if hasattr(best_model_LightGBM,"predict_proba"):
    proba_train_LightGBM = best_model_LightGBM.predict_proba(X_train)
elif hasattr(best_model_LightGBM,"decision_function"):
    proba_train_LightGBM=expit(best_model_LightGBM.decision_function(X_train))
    
pred_train_LightGBM = np.argmax(proba_train_LightGBM, axis=1)
pred_train_LightGBM=np.atleast_1d(pred_train_LightGBM)

# 测试集预测
if hasattr(best_model_LightGBM,"predict_proba"):
    proba_test_LightGBM = best_model_LightGBM.predict_proba(X_test)
elif hasattr(best_model_LightGBM,"decision_function"):
    proba_test_LightGBM=expit(best_model_LightGBM.decision_function(X_test))

pred_test_LightGBM = np.argmax(proba_test_LightGBM, axis=1)
pred_test_LightGBM=np.atleast_1d(pred_test_LightGBM)

# 计算LightGBM的评估指标

train_metrics_LightGBM = compute_metrics(y_train, proba_train_LightGBM)
test_metrics_LightGBM = compute_metrics(y_test, proba_test_LightGBM)

train_results['LightGBM'] = (y_train, proba_train_LightGBM, pred_train_LightGBM)  
test_results['LightGBM'] = (y_test, proba_test_LightGBM, pred_test_LightGBM) 
all_metrics_XGBoost = {'train': train_metrics_XGBoost, 'test': test_metrics_XGBoost} 
classes = sorted(np.unique(y_train))
XGBoost_metrics_list=[]
for dataset_name, metrics in all_metrics_XGBoost.items():
    # Per-class rows
    for cls in classes:
        row = {
            'model': 'XGBoost',
            'dataset': dataset_name,
            'class/avg': f'class_{cls}',
            'accuracy': metrics.get(f'accuracy_class_{cls}'), # Overall accuracy, repeated for context
            'auc': metrics.get(f'auc_class_{cls}'),
            'sensitivity': metrics.get(f'sensitivity_class_{cls}'),
            'specificity': metrics.get(f'specificity_class_{cls}'),
            'precision': metrics.get(f'precision_class_{cls}'),
            'f1': metrics.get(f'f1_class_{cls}'),
            '95%CI': metrics.get('95%CI')
        }
        XGBoost_metrics_list.append(row)

    # Averaged rows
    for avg in ['micro', 'macro', 'weighted']:
        row = {
            'model': 'XGBoost',
            'dataset': dataset_name,
            'class/avg': avg,
            'accuracy': metrics.get(f'accuracy_{avg}'), # Accuracy is overall, often reported with micro avg
            'auc': metrics.get(f'auc_{avg}'),
            'sensitivity': metrics.get(f'sensitivity_{avg}'),
            'specificity': metrics.get(f'specificity_{avg}'),
            'precision': metrics.get(f'precision_{avg}'),
            'f1': metrics.get(f'f1_{avg}'),
            '95%CI': metrics.get('95%CI')
        }
        XGBoost_metrics_list.append(row)

# 将 final_metrics_list 转换为 DataFrame
LightGBM_metrics_df = pd.DataFrame(LightGBM_metrics_list)

# 保存为 CSV 文件
LightGBM_metrics_df.to_csv('results/LightGBM_model_metrics.csv', index=False)

# 绘制LightGBM的ROC曲线
#plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_LightGBM, proba_test_LightGBM], 
         labels=['Train', 'Test'], title=f'Model: LightGBM (Macro/Micro AVG)',
         plot_type='averages')
plt.savefig(f'img/Rad_model_LightGBM_roc_averages.png', bbox_inches='tight')
plt.show()

# 绘制 per-class ROC 曲线
#plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_LightGBM, proba_test_LightGBM], 
         labels=['Train', 'Test'], title=f'Model: LightGBM (Per-Class)',
         plot_type='classes')
plt.savefig(f'img/Rad_model_LightGBM_roc_classes.png', bbox_inches='tight')
plt.show()
#绘制混淆矩阵
cm_test = confusion_matrix(y_test, pred_test_LightGBM)
labels = sorted(np.unique(y_test))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,annot_kws={"size": 40,"weight": "bold"},vmin=0,vmax=20,cbar_kws={'ticks': [0, 5, 10, 15, 20]})
#plt.title(f'Confusion Matrix for LightGBM',fontweight='bold',fontsize=30)
plt.ylabel('True label',fontweight='bold',fontsize=30)
plt.xlabel('Predicted label',fontweight='bold',fontsize=30)
plt.savefig(f'img/LightGBM_Confusion_Matrix.png')
plt.show()
plt.close()
#绘制DCA决策曲线
plot_multiclass_dca(y_test, proba_test_LightGBM, model_name='LightGBM', savepath="img/LightGBM_DCA.png")

#5.ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier
ExtraTrees_metrics_list = []
param_grid_ExtraTrees = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [4, 8, 10],
    'min_samples_leaf': [2, 4, 6]
}
ExtraTrees_model = ExtraTreesClassifier(random_state=42)

grid_search_ExtraTrees = GridSearchCV(ExtraTrees_model, param_grid_ExtraTrees, cv=10, scoring='roc_auc_ovr', n_jobs=-1)
grid_search_ExtraTrees.fit(X_train, y_train)
print(f"ExtraTrees Best params: {grid_search_ExtraTrees.best_params_}")
print(f"ExtraTrees Best score: {grid_search_ExtraTrees.best_score_}")
# 获取最优模型
best_model_ExtraTrees = grid_search_ExtraTrees.best_estimator_
# 创建模型字典
#model_names = ['ExtraTrees']  # 这里选择您想使用的模型名称（如 LR）
#models_dict = okcomp.comp1.create_clf_model(model_names)
#获取模型
#ExtraTrees_model = models_dict['ExtraTrees'] 

#best_model_ExtraTrees = ExtraTreesClassifier(n_estimators=200,max_depth=10,min_samples_split=10,min_samples_leaf=4,max_features=None)
#best_model_ExtraTrees.fit(X_train, y_train)
    
    # 训练集预测
if hasattr(best_model_ExtraTrees,"predict_proba"):
    proba_train_ExtraTrees = best_model_ExtraTrees.predict_proba(X_train)
elif hasattr(best_model_ExtraTrees,"decision_function"):
    proba_train_ExtraTrees=expit(best_model_ExtraTrees.decision_function(X_train))
    
pred_train_ExtraTrees = np.argmax(proba_train_ExtraTrees, axis=1) 
pred_train_ExtraTrees=np.atleast_1d(pred_train_ExtraTrees)
    

    # 测试集预测
if hasattr(best_model_ExtraTrees,"predict_proba"):
    proba_test_ExtraTrees = best_model_ExtraTrees.predict_proba(X_test)
elif hasattr(best_model_ExtraTrees,"decision_function"):
    proba_test_ExtraTrees=expit(best_model_ExtraTrees.decision_function(X_test))

pred_test_ExtraTrees = np.argmax(proba_test_ExtraTrees, axis=1)
pred_test_ExtraTrees=np.atleast_1d(pred_test_ExtraTrees)


train_metrics_ExtraTrees = compute_metrics(y_train, proba_train_ExtraTrees)
test_metrics_ExtraTrees = compute_metrics(y_test, proba_test_ExtraTrees)

train_results['ExtraTrees'] = (y_train, proba_train_ExtraTrees, pred_train_ExtraTrees)  
test_results['ExtraTrees'] = (y_test, proba_test_ExtraTrees, pred_test_ExtraTrees) 

all_metrics_ExtraTrees = {'train': train_metrics_ExtraTrees, 'test': test_metrics_ExtraTrees} 
classes = sorted(np.unique(y_train))
ExtraTrees_metrics_list=[]
for dataset_name, metrics in all_metrics_ExtraTrees.items():
    # Per-class rows
    for cls in classes:
        row = {
            'model': 'ExtraTrees',
            'dataset': dataset_name,
            'class/avg': f'class_{cls}',
            'accuracy': metrics.get(f'accuracy_class_{cls}'), # Overall accuracy, repeated for context
            'auc': metrics.get(f'auc_class_{cls}'),
            'sensitivity': metrics.get(f'sensitivity_class_{cls}'),
            'specificity': metrics.get(f'specificity_class_{cls}'),
            'precision': metrics.get(f'precision_class_{cls}'),
            'f1': metrics.get(f'f1_class_{cls}'),
            '95%CI': metrics.get(f'95%CI_class_{cls}')
        }
        ExtraTrees_metrics_list.append(row)

    # Averaged rows
    for avg in ['micro', 'macro', 'weighted']:
        row = {
            'model': 'ExtraTrees',
            'dataset': dataset_name,
            'class/avg': avg,
            'accuracy': metrics.get(f'accuracy_{avg}'), # Accuracy is overall, often reported with micro avg
            'auc': metrics.get(f'auc_{avg}'),
            'sensitivity': metrics.get(f'sensitivity_{avg}'),
            'specificity': metrics.get(f'specificity_{avg}'),
            'precision': metrics.get(f'precision_{avg}'),
            'f1': metrics.get(f'f1_{avg}'),
            '95%CI': metrics.get(f'95%CI_{avg}')
        }
        ExtraTrees_metrics_list.append(row)
# 将 final_metrics_list 转换为 DataFrame
ExtraTrees_metrics_df = pd.DataFrame(ExtraTrees_metrics_list)

# 保存为 CSV 文件
ExtraTrees_metrics_df.to_csv('results/ExtraTrees_model_metrics.csv', index=False)
# 绘制ExtraTrees的ROC曲线
#plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_ExtraTrees, proba_test_ExtraTrees], 
         labels=['Train', 'Test'], title=f'Model: ExtraTrees (Macro/Micro AVG)',
         plot_type='averages')
plt.savefig(f'img/Rad_model_ExtraTrees_roc_averages.png', bbox_inches='tight')
plt.show()

# 绘制 per-class ROC 曲线
#plt.figure(figsize=(8, 8))
draw_roc([np.array(y_train), np.array(y_test)], 
         [proba_train_ExtraTrees, proba_test_ExtraTrees], 
         labels=['Train', 'Test'], title=f'Model: ExtraTrees (Per-Class)',
         plot_type='classes')
plt.savefig(f'img/Rad_model_ExtraTrees_roc_classes.png', bbox_inches='tight')
plt.show()
#绘制混淆矩阵
cm_test = confusion_matrix(y_test, pred_test_ExtraTrees)
labels = sorted(np.unique(y_test))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,annot_kws={"size": 40,"weight": "bold"}
             ,vmin=0,vmax=20,cbar_kws={'ticks': [0, 5, 10, 15, 20]})
plt.title(f'Confusion Matrix for ExtraTrees',fontweight='bold',fontsize=30)
plt.ylabel('True label',fontweight='bold',fontsize=30)
plt.xlabel('Predicted label',fontweight='bold',fontsize=30)
plt.savefig(f'img/ExtraTrees_Confusion_Matrix.png')
plt.show()
plt.close()
#绘制DCA决策曲线
plot_multiclass_dca(y_test, proba_test_ExtraTrees, model_name='ExtraTrees', savepath="img/ExtraTrees_DCA.png")


     






