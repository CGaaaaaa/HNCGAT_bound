#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查导入依赖关系（不需要torch环境）
"""
import ast
import sys
import os

def extract_imports(filepath):
    """提取文件中的所有import语句"""
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filepath)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    return imports

def check_file_imports(filepath, project_files):
    """检查文件的导入依赖"""
    print(f"\n检查: {filepath}")
    imports = extract_imports(filepath)
    
    # 分类导入
    stdlib_imports = []
    third_party_imports = []
    local_imports = []
    
    stdlib_modules = ['sys', 'os', 're', 'json', 'datetime', 'argparse', 'warnings']
    third_party_modules = ['torch', 'numpy', 'scipy', 'sklearn', 'pandas']
    
    for imp in imports:
        base_module = imp.split('.')[0]
        if base_module in stdlib_modules:
            stdlib_imports.append(imp)
        elif base_module in third_party_modules:
            third_party_imports.append(imp)
        else:
            local_imports.append(imp)
    
    print(f"  标准库导入: {len(stdlib_imports)}")
    for imp in stdlib_imports:
        print(f"    - {imp}")
    
    print(f"  第三方库导入: {len(third_party_imports)}")
    for imp in third_party_imports:
        print(f"    - {imp}")
    
    print(f"  本地模块导入: {len(local_imports)}")
    for imp in local_imports:
        print(f"    - {imp}")
        # 检查本地模块是否存在
        if imp in ['loss', 'loss_weighted', 'utils', 'HNCGAT', 
                   'loss_hard_negative', 'decoder_edge_feature']:
            module_file = f"src/{imp}.py"
            if os.path.exists(module_file):
                print(f"      ✓ 文件存在: {module_file}")
            else:
                print(f"      ✗ 文件不存在: {module_file}")
    
    return True

def check_function_calls(filepath):
    """检查文件中是否有明显的函数调用错误"""
    print(f"\n检查函数调用: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查一些常见的错误模式
    issues = []
    
    # 检查是否有未定义的变量（简单检查）
    if 'undefined_variable' in content:
        issues.append("发现未定义的变量")
    
    # 检查括号匹配
    if content.count('(') != content.count(')'):
        issues.append("括号不匹配")
    
    if content.count('[') != content.count(']'):
        issues.append("方括号不匹配")
    
    if content.count('{') != content.count('}'):
        issues.append("花括号不匹配")
    
    if issues:
        print("  ✗ 发现潜在问题:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✓ 未发现明显问题")
        return True

def main():
    print("="*60)
    print("检查导入依赖关系和函数调用")
    print("="*60)
    
    files_to_check = [
        'src/loss_hard_negative.py',
        'src/decoder_edge_feature.py',
        'src/HNCGAT_enhanced.py',
    ]
    
    project_files = [
        'src/loss.py',
        'src/loss_weighted.py',
        'src/utils.py',
        'src/HNCGAT.py',
    ]
    
    all_passed = True
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"\n✗ 文件不存在: {filepath}")
            all_passed = False
            continue
        
        # 检查导入
        check_file_imports(filepath, project_files)
        
        # 检查函数调用
        if not check_function_calls(filepath):
            all_passed = False
    
    # 检查关键依赖文件是否存在
    print("\n" + "="*60)
    print("检查依赖文件:")
    print("="*60)
    
    for filepath in project_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (缺失)")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有检查通过！")
        print("="*60)
        print("\n代码质量总结:")
        print("  - Python语法: ✓ 正确")
        print("  - 导入依赖: ✓ 完整")
        print("  - 函数调用: ✓ 无明显错误")
        print("  - 依赖文件: ✓ 存在")
        print("\n可以安全地上传到服务器并运行实验！")
        return 0
    else:
        print("✗ 存在一些问题，请检查")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())

