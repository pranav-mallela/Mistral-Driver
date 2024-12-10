# Mistral-Driver

Through this project, we explore the potential of modern LLMs in end-to-end planning for autonomous driving.

First, request access to the Mistral-7B model at this Hugging Face [link](https://huggingface.co/mistralai/Mistral-7B-v0.3) and obtain an access token. Then create a file by the name `.env`, and add the following line:

`export HF_TOKEN=<YOUR_ACCESS_TOKEN>`

Now, in your preferred environment, run the command: `pip install transformers datasets evaluate peft accelerate trl bitsandbytes einops xformers wandb gradio sentencepiece`

To fine-tune Mistral-7B and generate test results: run `python3 driving_finetuning.py`

To run the out-of-box Mistral-7B and generate test results: run `python3 driving_outofthebox.py`

Results are generated in `results_finetuning.json` and `results_outofthebox.json` respectively.

To generate insights based on results,

* For the fine-tuned model, run `python3 analyze_finetuning.py`
* For the out-of-box model, run `python3 analyze_outofthebox.py`
