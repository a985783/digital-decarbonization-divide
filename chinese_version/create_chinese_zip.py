"""
创建中文版提交包
==================
打包中文版论文及所有必要文件。
"""

import zipfile
import os

OUTPUT_ZIP = 'chinese_submission.zip'

# 要包含的文件（相对路径）
FILES_TO_INCLUDE = {
    # 核心文档
    'paper.tex': 'paper.tex',
    'README.md': 'README.md',
    'requirements.txt': 'requirements.txt',
    
    # 数据
    'data/wdi_expanded_raw.csv': 'data/wdi_expanded_raw.csv',
    'data/clean_data_v3_imputed.csv': 'data/clean_data_v3_imputed.csv',
    
    # 脚本
    'scripts/solve_wdi_v4_expanded_zip.py': 'scripts/solve_wdi_v4_expanded_zip.py',
    'scripts/impute_mice.py': 'scripts/impute_mice.py',
    'scripts/lasso_selection.py': 'scripts/lasso_selection.py',
    'scripts/dml_causal_v2.py': 'scripts/dml_causal_v2.py',
    'scripts/xgboost_shap_v3.py': 'scripts/xgboost_shap_v3.py',
    'scripts/mechanism_check.py': 'scripts/mechanism_check.py',
    
    # 结果
    'results/figures/lasso_path_v3.png': 'results/figures/lasso_path_v3.png',
    'results/figures/shap_dependence_v3.png': 'results/figures/shap_dependence_v3.png',
    'results/figures/shap_summary_v3.png': 'results/figures/shap_summary_v3.png',
}

def create_zip():
    print(f"创建 {OUTPUT_ZIP}...")
    
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, dst in FILES_TO_INCLUDE.items():
            if os.path.exists(src):
                print(f"  添加: {src} -> {dst}")
                zf.write(src, dst)
            else:
                print(f"⚠️ 警告: 缺少文件 {src}")
    
    print(f"\n✓ 成功创建 {OUTPUT_ZIP}")
    print("  可用于提交/上传。")

if __name__ == "__main__":
    create_zip()
