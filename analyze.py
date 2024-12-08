

# we analyze key metrics from the results

import json
from collections import Counter

# Load the JSON file
file_path = "results_finetuning.json"  # Replace with your actual JSON file path
with open(file_path, 'r') as file:
    data = json.load(file)

# Labels of interest
labels = ["SLOW", "ACCELERATE", "MAINTAIN", "STOPPED", "LEFT", "RIGHT", "REVERSE", "OTHER"]

# Initialize counters
METRICS_BY_LABEL = {
    "SLOW":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "STOPPED":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "ACCELERATE":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "MAINTAIN":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "LEFT":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "RIGHT":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "REVERSE":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "OTHER":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    }
}

BUCKET_METRICS_BY_LABEL = {
    "SLOW":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "MAINTAIN":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "TURN":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "REVERSE":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    },
    "OTHER":{
        "PREDICTED_COUNT": 0,
        "GROUND_TRUTH_COUNT": 0,
        "CORRECTLY_PREDICTED_COUNT": 0,
        "INCORRECTLY_PREDICTED_COUNT": 0
    }
}


# Iterate through the JSON objects
for entry in data:
    ground_truth = entry["ground_truth"]
    ground_truth_bucket = entry["ground_truth_bucket"]
    prediction = entry["prediction"]
    prediction_bucket = entry["prediction_bucket"]


    METRICS_BY_LABEL[ground_truth]["GROUND_TRUTH_COUNT"] += 1
    BUCKET_METRICS_BY_LABEL[ground_truth_bucket]["GROUND_TRUTH_COUNT"] += 1

    METRICS_BY_LABEL[prediction]["PREDICTED_COUNT"] += 1
    BUCKET_METRICS_BY_LABEL[prediction_bucket]["PREDICTED_COUNT"] += 1


    if (prediction == ground_truth):
        METRICS_BY_LABEL[prediction]["CORRECTLY_PREDICTED_COUNT"] += 1
    else:
        METRICS_BY_LABEL[prediction]["INCORRECTLY_PREDICTED_COUNT"] += 1

    if (prediction_bucket == ground_truth_bucket):
        BUCKET_METRICS_BY_LABEL[prediction_bucket]["CORRECTLY_PREDICTED_COUNT"] += 1
    else:
        BUCKET_METRICS_BY_LABEL[prediction_bucket]["INCORRECTLY_PREDICTED_COUNT"] += 1



print("---------------- Reporting Ground Truth Count, Exact-Match")
total = 0
for label, metrics in METRICS_BY_LABEL.items():
    ground_truth = metrics["GROUND_TRUTH_COUNT"]
    total += ground_truth
    print(f"{label}: {ground_truth} Ground Truth examples")
print(f"TOTAL: {total} Ground Truth examples")


print("---------------- Reporting Ground Truth Count, Close-Match")
total = 0
for label, metrics in BUCKET_METRICS_BY_LABEL.items():
    ground_truth = metrics["GROUND_TRUTH_COUNT"]
    total += ground_truth
    print(f"{label}: {ground_truth} Ground Truth examples")
print(f"TOTAL: {total} Ground Truth examples")



print("---------------- Reporting Per-label Accuracy, Exact-Match")
for label, metrics in METRICS_BY_LABEL.items():
    ground_truth = metrics["GROUND_TRUTH_COUNT"]
    correctly_predicted = metrics["CORRECTLY_PREDICTED_COUNT"]
    if ground_truth > 0:
        ratio = correctly_predicted / ground_truth
        print(f"{label}: {ratio} (Correctly Predicted / Ground Truth)")
    else:
        print(f"{label}: No Ground Truth examples")


print("---------------- Reporting Per-label Exact-Match Precision")
for label, metrics in METRICS_BY_LABEL.items():
    predicted = metrics["PREDICTED_COUNT"]
    correctly_predicted = metrics["CORRECTLY_PREDICTED_COUNT"]
    if predicted > 0:
        ratio = correctly_predicted / predicted
        print(f"{label}: {ratio} (Correctly Predicted / Total Predicted)")
    else:
        print(f"{label}: Never Predicted")


print("---------------- Reporting Per-label Accuracy, Close-Match")
for label, metrics in BUCKET_METRICS_BY_LABEL.items():
    ground_truth = metrics["GROUND_TRUTH_COUNT"]
    correctly_predicted = metrics["CORRECTLY_PREDICTED_COUNT"]
    if ground_truth > 0:
        ratio = correctly_predicted / ground_truth
        print(f"{label}: {ratio} (Correctly Predicted / Ground Truth)")
    else:
        print(f"{label}: No Ground Truth examples")


print("---------------- Reporting Per-label Close-Match Precision")
for label, metrics in BUCKET_METRICS_BY_LABEL.items():
    predicted = metrics["PREDICTED_COUNT"]
    correctly_predicted = metrics["CORRECTLY_PREDICTED_COUNT"]
    if predicted > 0:
        ratio = correctly_predicted / predicted
        print(f"{label}: {ratio} (Correctly Predicted / Total Predicted)")
    else:
        print(f"{label}: Never Predicted")




