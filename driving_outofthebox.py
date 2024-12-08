import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import pipeline
from dotenv import load_dotenv
import json

"""
Process Data
""" 
# ordered in priority (earlier labels supercede later ones)
# labels and their key-phrases
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

BUCKET_MAPPING = {
    "MAINTAIN": "MAINTAIN",
    "ACCELERATE": "MAINTAIN",
    "STOPPED": "SLOW",
    "SLOW": "SLOW",
    "RIGHT": "TURN",
    "LEFT": "TURN",
    "REVERSE": "REVERSE",
    "OTHER": "OTHER"
}

def check_label_match(full_string, label_key):
    if label_key == "OTHER":
        return True

    if label_key == "LEFT" or label_key == "RIGHT":
        match = True
        for key_word_set in LABELS[label_key]:
            set_match = False
            for key_word in key_word_set:
                if key_word in full_string:
                    set_match = True
            match = (match and set_match)
        if match:
            return True

    else:
        for key_word in LABELS[label_key]:
            if key_word in full_string:
                return True

    return False

def get_label(full_string):
    for k in LABELS.keys():
        if (check_label_match(full_string, k)):
            return k

def clean_reason(reason):
    words_to_remove = ["because", "to", "since", "as", "due", "for"]
    # Split the reason into words
    words = reason.split()
    # Check if the first word is one of the words to remove
    if words and words[0] in words_to_remove:
        # Remove the first word
        words = words[1:]
    if words and words[0] in words_to_remove:
        # Remove the first word
        words = words[1:]
    # Join the words back into a string
    return " ".join(words)

def load_examples():

    df = pd.read_csv('dataset.csv')

    df = df.iloc[:, 3:] # remove first 3 cols

    examples = []

    for index, row in df.iterrows():

        # Iterate through columns in steps of 2 (pairing columns)
        for i in range(0, len(row), 2):  # Start from 0, step by 2 to get pairs
            first_value = row[i]
            second_value = row[i + 1] if (i + 1) < len(row) else None  # Handle case where second value might be missing

            # Check if the first value is NaN
            if pd.isna(first_value):
                break  # Stop processing once the first value in the pair is NaN

            if second_value == None or pd.isna(second_value):
                break

            if not isinstance(first_value, (int, float)):
                # Append the pair to the processed list
                examples.append((first_value, second_value))

    return examples

def preprocess_examples(raw_examples):
    examples = []
    others = 0
    for e in raw_examples:
        examples.append((get_label(e[0]), clean_reason(e[1])))
        # if examples[-1][0] == "OTHER":
        #     print(e[0], " = OTHER")
        #     others += 1

    # print("Num. OTHER:", others)
    return examples

def extract_output_lbl(answer):
    for lbl in LABELS:
        if answer.find(lbl) != -1:
            return lbl
    return "OTHER"

##################################
"""
Generate Prompt
"""
def get_prompt(example, few_shot_examples=None, system_prompt="You are an AI driving assistant that chooses a single instruction from [MAINTAIN,SLOW,ACCELERATE,RIGHT,LEFT,REVERSE,OTHER] based on a given driving scenario.\n"):
  examples_segment = ""
  if few_shot_examples is not None:
    examples_segment = "Here are some examples of how to perform this task:\n\n"
    for e in few_shot_examples:
      examples_segment += f'''Situation: {e["description"]}\n{e["label"]}\n'''

  user_prompt = f'''\nChoose one instruction from [MAINTAIN,SLOW,ACCELERATE,RIGHT,LEFT,REVERSE,OTHER] for this situation, and give your one-word answer after: {example["description"]}'''
  return [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": examples_segment + user_prompt},
  ]
###############################
"""
Calculate Accuracy
"""
# def get_acc(predicted_labels, true_labels):
#   pred_arr = np.array([lbl.replace(" ", "") for lbl in predicted_labels])
#   true_arr = np.array([lbl.replace(" ", "") for lbl in true_labels])
#   return (pred_arr == true_arr).mean()

# def get_bucket_acc(predicted_labels, true_labels):
#   predicted_labels = [lbl.replace(" ", "") for lbl in predicted_labels]
#   predicted_labels = ["MAINTAIN" if lbl in ["MAINTAIN", "ACCELERATE"] else lbl for lbl in predicted_labels]
#   predicted_labels = ["SLOW" if lbl in ["STOPPED", "SLOW"] else lbl for lbl in predicted_labels]
#   predicted_labels = ["OTHER" if lbl not in ["MAINTAIN", "SLOW", "RIGHT", "LEFT"] else lbl for lbl in predicted_labels]

#   true_labels = [lbl.replace(" ", "") for lbl in true_labels]
#   true_labels = ["MAINTAIN" if lbl in ["MAINTAIN", "ACCELERATE"] else lbl for lbl in true_labels]
#   true_labels = ["SLOW" if lbl in ["STOPPED", "SLOW"] else lbl for lbl in true_labels]
#   true_labels = ["OTHER" if lbl not in ["MAINTAIN", "SLOW", "RIGHT", "LEFT"] else lbl for lbl in true_labels]

#   return get_acc(predicted_labels, true_labels) # now checks for MAINTAIN, SLOW, RIGHT, LEFT, OTHER

def compute_accuracy(results):
    count = 0
    bucket_count = 0
    for result in results:
        # Check if prediction matches the label
        if result['prediction'] == result['ground_truth']: # first 111 chars are the original prompt & don't contain new output
            count += 1

        # Check if pred vs ground truth buckets match
        if result['prediction_bucket'] == result['ground_truth_bucket']:
            bucket_count += 1

    return ((count / len(results)),(bucket_count / len(results)))

print("Results saved to results.json")

def main():
    # Load Data
    raw_examples = load_examples()
    examples = preprocess_examples(raw_examples)

    df = pd.DataFrame(examples, columns=["label", "description"])
    dataset = Dataset.from_pandas(df)
    shuffled_dataset = dataset.shuffle(seed=42) # shuffle to account for implicit order in existing scenario-reason pairs
    sample_size = int(0.50 * len(shuffled_dataset)) # choose 50% of the dataset for fine-tuning in under 2 hours on gpu_mig40
    sampled_dataset = shuffled_dataset.select(range(sample_size))
    train_test_split = sampled_dataset.train_test_split(test_size=0.3, seed=42)
    validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'validation': validation_test_split['train'],
        'test': validation_test_split['test']
    })

    # Initialize Model
    load_dotenv()
    llm_generator = pipeline('text-generation', model="mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")

    # Few Shot Examples for Prompt
    few_shot_examples = [
        ("MAINTAIN", "traffic is clear."),
        ("STOPPED", "it turns to the right."),
        ("ACCELERATE", "the light has turned green and traffic is flowing."),
        ("MAINTAIN", "traffic flows freely."),
        ("LEFT", "get around a slower car in front of it."),
        ("MAINTAIN", "traffic moves freely."),
        ("RIGHT", "a car in the neighboring lane entering the car's lane."),
        ("MAINTAIN", "there are no nearby cars in its lane."),
        ("STOPPED", "the light is red."),
        ("ACCELERATE", "the light turned green."),
        ("SLOW", "traffic in front of it is stopped."),
        ("MAINTAIN", "slow traffic in front of it."),
        ("LEFT", "make a left turn.")
    ]

    fs = pd.DataFrame(few_shot_examples, columns=["label", "description"])
    fs_dataset = Dataset.from_pandas(fs)

    # Test Model on Data
    results = []
    for mode in ["test"]: # ["train", "validation", test"]
        data = dataset_dict[mode]
        predictions = []
        for example in data:
            prompt = get_prompt(example, fs_dataset)
            # print(prompt[1]["content"])
            full_output = llm_generator(prompt, max_new_tokens=200)[0]["generated_text"]
            for elt in full_output:
                if elt['role'] == 'assistant':
                    final_answer = elt['content']
                    extract_output_lbl(final_answer)
                    # Store the prediction and ground truth
                    results.append({
                        "description": example['description'],
                        "prediction": final_answer,
                        "ground_truth": example['label'],
                        "prediction_bucket": BUCKET_MAPPING[final_answer],
                        "ground_truth_bucket": BUCKET_MAPPING[example['label']]
                    })
                    predictions.append(final_answer)
                    # print(f'''{final_answer} for SITUATION: {example["description"]}''')
        
        # Calculate Accuracy
        # acc = get_acc(predictions, data["label"])
        # bucket_acc = get_bucket_acc(predictions, data["label"])
        # print("Accuracy: ", acc)
        # print("Bucket Accuracy: ", bucket_acc)
        accuracy, bucket_accuracy = compute_accuracy(results)

        print("Exact Accuracy:", accuracy)
        print("Bucket Accuracy:", bucket_accuracy)

        # Save Results
        # with open(f'predictions_{mode}.txt', 'w') as f:
        #     for i, pred in enumerate(predictions):
        #         f.write(f"{pred} : {data['label'][i]}\n")
        with open("results_finetuning.json", "w") as file:
            json.dump(results, file, indent=4)

        print("Results saved to results.json")

if __name__ == "__main__":
   main()