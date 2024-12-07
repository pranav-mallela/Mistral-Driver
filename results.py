import numpy as np

LABELS = {
    "STOPPED": (["not moving", "stopped", "still", "sitting", "stop", "isn't moving", "not moving", "stops", "parked","stationary"]),
    "REVERSE": (["reverse", "reverses", "reversing"]),
    "ACCELERATE": (["accelerates", "resumes", "picks up", "speeding up", "speeds up", "accelerate", "accelerating"]),
    "RIGHT": (["merge", "turn", "turns", "veer", "makes", "veers", "steers", "steering", "merges",  "moves", "moving", "shifting", "switching", "merging", "turning"],["right"]), # needs a word in BOTH to match
    "LEFT": (["merge", "turn", "turns", "veer", "makes", "veers", "steers", "steering", "merges",  "moves", "moving", "shifting", "switching", "merging", "turning"],["left"]), # needs a word in BOTH to match
    "MAINTAIN": (["travelling down", "going fast", "continues", "continue", "steady", "moves down", "moves forward", "moving down", "moving forward", "drives slowly", "stays", "inches", "drives forward", "steadily", "driving slowly", "driving", "drives", "driving forward", "maintains"]),
    "SLOW": (["slows", "slowing", "brakes", "braking", "slow", "is stopping"]),
    "OTHER": ([])
}

"""
Calculate Accuracy
"""
def get_acc(predicted_labels, true_labels):
  pred_arr = np.array([lbl.replace(" ", "") for lbl in predicted_labels])
  true_arr = np.array([lbl.replace(" ", "") for lbl in true_labels])
  return (pred_arr == true_arr).mean()

def get_bucket_acc(predicted_labels, true_labels):
  predicted_labels = [lbl.replace(" ", "") for lbl in predicted_labels]
  predicted_labels = ["MAINTAIN" if lbl in ["MAINTAIN", "ACCELERATE"] else lbl for lbl in predicted_labels]
  predicted_labels = ["SLOW" if lbl in ["STOPPED", "SLOW"] else lbl for lbl in predicted_labels]
  predicted_labels = ["OTHER" if lbl not in ["MAINTAIN", "SLOW", "RIGHT", "LEFT"] else lbl for lbl in predicted_labels]

  true_labels = [lbl.replace(" ", "") for lbl in true_labels]
  true_labels = ["MAINTAIN" if lbl in ["MAINTAIN", "ACCELERATE"] else lbl for lbl in true_labels]
  true_labels = ["SLOW" if lbl in ["STOPPED", "SLOW"] else lbl for lbl in true_labels]
  true_labels = ["OTHER" if lbl not in ["MAINTAIN", "SLOW", "RIGHT", "LEFT"] else lbl for lbl in true_labels]

  return get_acc(predicted_labels, true_labels) # now checks for MAINTAIN, SLOW, RIGHT, LEFT, OTHER

def extract_output_lbl(answer):
    for lbl in LABELS:
        if answer.find(lbl) != -1:
            return lbl
    return "OTHER"

def main():
    predictions, labels = [], []
    with open('predictions_test.txt', 'r') as f:
        for line in f:
            line = line.strip()
            pred, lbl = line.split(':', 1)
            predictions.append(extract_output_lbl(pred))
            labels.append(lbl)
    
    print("Exact Accuracy: ", get_acc(predictions, labels))
    print("Bucket Accuracy: ", get_bucket_acc(predictions, labels))

if __name__ == "__main__":
   main()
