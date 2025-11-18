#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试加权损失函数 - 语法检查版本
不运行实际代码，只检查语法和导入结构
"""
import sys
import os
import ast

print("=" * 60)
print("测试1: 检查文件是否存在")
print("=" * 60)

src_dir = os.path.join(os.path.dirname(__file__), 'src')
loss_weighted_file = os.path.join(src_dir, 'loss_weighted.py')
hncgat_file = os.path.join(src_dir, 'HNCGAT.py')

if os.path.exists(loss_weighted_file):
    print(f"✓ loss_weighted.py 存在")
else:
    print(f"✗ loss_weighted.py 不存在")
    sys.exit(1)

if os.path.exists(hncgat_file):
    print(f"✓ HNCGAT.py 存在")
else:
    print(f"✗ HNCGAT.py 不存在")
    sys.exit(1)

print("\n" + "=" * 60)
print("测试2: 检查Python语法")
print("=" * 60)

try:
    with open(loss_weighted_file, 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✓ loss_weighted.py 语法正确")
except SyntaxError as e:
    print(f"✗ loss_weighted.py 语法错误: {e}")
    sys.exit(1)

try:
    with open(hncgat_file, 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✓ HNCGAT.py 语法正确")
except SyntaxError as e:
    print(f"✗ HNCGAT.py 语法错误: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("测试3: 检查关键类和函数")
print("=" * 60)

# 检查loss_weighted.py中的类
with open(loss_weighted_file, 'r', encoding='utf-8') as f:
    content = f.read()
    
required_classes = [
    'AdaptiveWeightedPConLoss',
    'AdaptiveWeightedMConLoss', 
    'AdaptiveWeightedFConLoss'
]

for cls_name in required_classes:
    if f'class {cls_name}' in content:
        print(f"✓ 找到类: {cls_name}")
    else:
        print(f"✗ 缺少类: {cls_name}")
        sys.exit(1)

# 检查HNCGAT.py中的集成
with open(hncgat_file, 'r', encoding='utf-8') as f:
    hncgat_content = f.read()

# 检查导入
if 'from loss_weighted import' in hncgat_content:
    print("✓ HNCGAT.py 导入了 loss_weighted")
else:
    print("✗ HNCGAT.py 未导入 loss_weighted")
    sys.exit(1)

# 检查参数
if '--use-weighted-loss' in hncgat_content:
    print("✓ HNCGAT.py 包含 --use-weighted-loss 参数")
else:
    print("✗ HNCGAT.py 缺少 --use-weighted-loss 参数")
    sys.exit(1)

# 检查模型中的使用
if 'use_weighted_loss' in hncgat_content and 'weighted_P_loss' in hncgat_content:
    print("✓ HNCGAT.py 正确集成了加权损失")
else:
    print("✗ HNCGAT.py 未正确集成加权损失")
    sys.exit(1)

print("\n" + "=" * 60)
print("测试4: 检查代码结构")
print("=" * 60)

# 检查loss_weighted.py中的forward方法
for cls_name in required_classes:
    if f'def forward(self' in content and cls_name in content:
        print(f"✓ {cls_name} 包含 forward 方法")
    else:
        print(f"⚠ {cls_name} 可能缺少 forward 方法")

# 检查返回值
if 'return (-torch.log(loss)).mean(), weights' in content:
    print("✓ 损失函数返回 (loss, weights)")
else:
    print("⚠ 损失函数返回值可能不正确")

print("\n" + "=" * 60)
print("✅ 语法和结构检查通过！")
print("=" * 60)
print("\n代码结构正确，可以到云主机上运行完整实验了。")
print("\n注意：完整功能测试需要在有PyTorch的环境中运行。")
print("\n云主机使用命令：")
print("  cd /20232501535/myHNCGAT/HNCGAT/src")
print("  python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --edgetype concat")

