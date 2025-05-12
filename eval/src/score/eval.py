import argparse
import json
import os

from scipy.stats import spearmanr, pearsonr

arg = argparse.ArgumentParser()
arg.add_argument("--pred_json", type=str, default="pred.json")
arg.add_argument("--gt_json", type=str, default="../ViDA-MIPI-dataset/val/ViDA-Bench/ground_truth/bench_metadata.json")

if __name__ == "__main__":
    args = arg.parse_args()
    print(args.pred_json)
    print(args.gt_json)
    with open(args.pred_json, "r") as f:
        pred = json.load(f)
    with open(args.gt_json, "r") as f:
        gt = json.load(f)
    
    assert len(pred) == len(gt), "The length of predicted and ground truth is not the same"
    
    pred = sorted(pred, key=lambda x: x["id"])
    gt = sorted(gt, key=lambda x: x["id"])

    pred_mos = [item["pred_mos"] for item in pred]
    gt_mos = [item["mos"] for item in gt]
    
    spearmanr_score = spearmanr(pred_mos, gt_mos)
    print(f"Spearmanr score: {spearmanr_score}")    
    pearsonr_score = pearsonr(pred_mos, gt_mos)
    print(f"Pearsonr score: {pearsonr_score}")
    overall_score = (spearmanr_score + pearsonr_score) / 2
    print(f"Overall score: {overall_score}")