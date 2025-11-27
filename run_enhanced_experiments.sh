#!/bin/bash
# 增强版HNCGAT实验脚本
# 
# 实验配置：
# 1. Baseline (原HNCGAT)
# 2. Weighted Loss only
# 3. Weighted + Diffusion (当前最好)
# 4. Weighted + Diffusion + Hard Negative Mining (新)
# 5. Weighted + Diffusion + Edge Features (新)
# 6. Weighted + Diffusion + Hard Negative + Edge Features (全部创新)

cd /Users/consingliu/Desktop/des/sjm/myessay/HNCGAT_combine_diffusion/src

echo "=========================================="
echo "增强版HNCGAT实验 - 30% 训练比例"
echo "=========================================="

# 实验1: Baseline (原HNCGAT，无任何创新)
echo ""
echo "实验1: Baseline (原HNCGAT)"
nohup python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --gpu 0 \
    --n_epochs 1000 \
    --n_runs 5 \
    > ../logs/exp1_baseline_30pct.log 2>&1 &
echo "已启动，日志: logs/exp1_baseline_30pct.log"
sleep 5

# 实验2: Weighted Loss only
echo ""
echo "实验2: Weighted Loss only"
nohup python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --gpu 0 \
    --n_epochs 1000 \
    --n_runs 5 \
    > ../logs/exp2_weighted_30pct.log 2>&1 &
echo "已启动，日志: logs/exp2_weighted_30pct.log"
sleep 5

# 实验3: Weighted + Diffusion (当前最好的配置)
echo ""
echo "实验3: Weighted + Diffusion (当前最好)"
nohup python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --use-diffusion \
    --diffusion-K 3 \
    --diffusion-alpha 0.2 \
    --diffusion-beta 0.5 \
    --gpu 0 \
    --n_epochs 1000 \
    --n_runs 5 \
    > ../logs/exp3_weighted_diffusion_30pct.log 2>&1 &
echo "已启动，日志: logs/exp3_weighted_diffusion_30pct.log"
sleep 5

# 实验4: Weighted + Diffusion + Hard Negative Mining (新创新1)
echo ""
echo "实验4: Weighted + Diffusion + Hard Negative Mining (新)"
nohup python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --use-diffusion \
    --diffusion-K 3 \
    --diffusion-alpha 0.2 \
    --diffusion-beta 0.5 \
    --use-hard-negative \
    --hard-negative-ratio 0.3 \
    --hard-negative-weight 2.0 \
    --gpu 0 \
    --n_epochs 1000 \
    --n_runs 5 \
    > ../logs/exp4_weighted_diffusion_hardneg_30pct.log 2>&1 &
echo "已启动，日志: logs/exp4_weighted_diffusion_hardneg_30pct.log"
sleep 5

# 实验5: Weighted + Diffusion + Edge Features (新创新2)
echo ""
echo "实验5: Weighted + Diffusion + Edge Features (新)"
nohup python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --use-diffusion \
    --diffusion-K 3 \
    --diffusion-alpha 0.2 \
    --diffusion-beta 0.5 \
    --use-edge-features \
    --gpu 0 \
    --n_epochs 1000 \
    --n_runs 5 \
    > ../logs/exp5_weighted_diffusion_edgefeat_30pct.log 2>&1 &
echo "已启动，日志: logs/exp5_weighted_diffusion_edgefeat_30pct.log"
sleep 5

# 实验6: 全部创新 (Weighted + Diffusion + Hard Negative + Edge Features)
echo ""
echo "实验6: 全部创新 (Weighted + Diffusion + Hard Negative + Edge Features)"
nohup python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --use-diffusion \
    --diffusion-K 3 \
    --diffusion-alpha 0.2 \
    --diffusion-beta 0.5 \
    --use-hard-negative \
    --hard-negative-ratio 0.3 \
    --hard-negative-weight 2.0 \
    --use-edge-features \
    --gpu 0 \
    --n_epochs 1000 \
    --n_runs 5 \
    > ../logs/exp6_all_innovations_30pct.log 2>&1 &
echo "已启动，日志: logs/exp6_all_innovations_30pct.log"

echo ""
echo "=========================================="
echo "所有实验已启动！"
echo "=========================================="
echo ""
echo "查看进度："
echo "  tail -f ../logs/exp1_baseline_30pct.log"
echo "  tail -f ../logs/exp4_weighted_diffusion_hardneg_30pct.log"
echo "  tail -f ../logs/exp5_weighted_diffusion_edgefeat_30pct.log"
echo "  tail -f ../logs/exp6_all_innovations_30pct.log"
echo ""
echo "查看所有运行的进程："
echo "  ps aux | grep HNCGAT_enhanced"
echo ""

