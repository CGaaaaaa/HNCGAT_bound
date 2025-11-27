#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语法检查脚本 - 不需要torch环境，只检查Python语法
"""
import sys
import py_compile
import os

def check_syntax(filepath):
    """检查单个文件的语法"""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    print("="*60)
    print("检查Python语法（不需要torch环境）")
    print("="*60)
    
    files_to_check = [
        'src/loss_hard_negative.py',
        'src/decoder_edge_feature.py',
        'src/HNCGAT_enhanced.py',
    ]
    
    all_passed = True
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"\n✗ 文件不存在: {filepath}")
            all_passed = False
            continue
        
        print(f"\n检查: {filepath}")
        success, error = check_syntax(filepath)
        
        if success:
            print(f"  ✓ 语法正确")
            # 检查文件大小
            size = os.path.getsize(filepath)
            print(f"  - 文件大小: {size/1024:.1f} KB")
            # 统计行数
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            print(f"  - 代码行数: {lines} 行")
        else:
            print(f"  ✗ 语法错误:")
            print(f"    {error}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有文件语法检查通过！")
        print("="*60)
        print("\n下一步：")
        print("1. 将这些文件上传到服务器")
        print("2. 在服务器上运行: python test_enhanced_code.py")
        print("3. 如果测试通过，运行: bash run_enhanced_experiments.sh")
        return 0
    else:
        print("✗ 有文件存在语法错误，请修复后再试")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())

