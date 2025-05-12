import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, help='Path to the data file')
parser.add_argument('--gt_file', type=str, default='../ViDA-MIPI-dataset/val/ViDA-Bench/ground_truth/mcq.json', help='Path to the ground truth file')
args = parser.parse_args()

# Load the data
with open(args.pred_file, 'r') as f:
    data = json.load(f)
with open(args.gt_file, 'r') as f:
    gt_data = json.load(f)

assert len(data) == len(gt_data), "The length of the predicted data and the ground truth data are not the same."
data = sorted(data, key=lambda x: x["id"])
gt_data = sorted(gt_data, key=lambda x: x["id"])
# Initialize counts and correct predictions
concern_counts = {}
concern_correct = {}

# Iterate over the data to populate counts and correct predictions
for pred_item, gt_item in zip(data, gt_data):
    c = gt_item['concern']
    correct = gt_item['correct_ans']
    # Ensure that 'pred_ans' is stripped of any extra characters like '.' and mapped correctly.
    pred_index = ord(pred_item['pred_ans'].strip()[0]) - ord('A')

    pred = pred_item['candidates'][pred_index]  # Map 'A', 'B', 'C', etc. to the correct answer

    # Update concern counts
    concern_counts[c] = concern_counts.get(c, 0) + 1
    concern_correct[c] = concern_correct.get(c, 0) + (1 if correct == pred else 0)

# Calculate accuracy for each type and concern
concern_accuracy = {c: concern_correct[c] / concern_counts[c] for c in concern_counts}

# Calculate overall accuracy
total_correct = sum(concern_correct.values())
total_count = sum(concern_counts.values())
overall_accuracy = total_correct / total_count if total_count > 0 else 0

# Print the results
print("Concern Counts:", concern_counts)
print("Concern Accuracy:", concern_accuracy)
print("Overall Accuracy:", overall_accuracy)