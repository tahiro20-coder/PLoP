import torch
import random
from datasets import load_dataset
import string

# Only keep the minimal set of functions needed for the simplified main


def load_gsm8k_problems(num_samples=200):
    """
    Load problems from the GSM8K dataset.
    Args:
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    Remark: For now, we only keep the questions. Answers can be included in the samples as well.
    """
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    import requests, json
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    problems = [json.loads(line)['question'] for line in lines[:num_samples*2]]
    return random.sample(problems, min(num_samples, len(problems)))

def load_mmlu_problems(subject="logical_fallacies", num_samples=200):
    """
    Load problems from the MMLU dataset.
    Args:
        subject: Subject to load.
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    """
    dataset = load_dataset("cais/mmlu", subject, split="test")
    problems = []
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        problem = f"{question}\n" + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        problems.append(problem)
    return random.sample(problems, min(num_samples, len(problems)))

def load_humaneval_problems(num_samples=200):
    """
    Load problems from the HumanEval dataset.
    Args:
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    """
    dataset = load_dataset("openai_humaneval", split="test")
    problems = [f"# Write a Python function\n{item['prompt']}" for item in dataset]
    return random.sample(problems, min(num_samples, len(problems)))

def load_medqa_problems(num_samples=200, split="train"):
    """
    Load problems from the MedQA-USMLE dataset.
    
    This dataset contains medical multiple-choice questions from the US Medical 
    Licensing Examination (USMLE). Each question has 4-5 answer options.
    
    Args:
        num_samples: Number of samples to load.
        split: Dataset split to use ('train', 'validation', or 'test').
    
    Returns:
        List of problems formatted as question + answer choices.
    
    Note:
        The function uses the GBaker/MedQA-USMLE-4-options dataset from Hugging Face,
        which is a cleaned version with 4 options per question.
    """
    try:
        # Load the MedQA dataset from Hugging Face
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        
        problems = []
        for item in dataset:
            question = item["question"]
            
            # Get the answer options
            if isinstance(item["options"], dict):
                options = item["options"]
                choices_text = "\n".join([f"{key}. {value}" for key, value in sorted(options.items())])
            elif isinstance(item["options"], list):
                options = item["options"]
                choices_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            else:
                continue
            
            problem = f"{question}\n{choices_text}"
            problems.append(problem)
        
        return random.sample(problems, min(num_samples, len(problems)))
    
    except Exception as e:
        print(f"Error loading MedQA dataset: {e}")
        # Try fallback
        try:
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa", split=split)
            problems = []
            for item in dataset:
                question = item["question"]
                choices = item["choices"]
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                problem = f"{question}\n{choices_text}"
                problems.append(problem)
            return random.sample(problems, min(num_samples, len(problems)))
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return []

def prepare_batch(problems, tokenizer, max_length=256):
    """
    Prepare a batch of problems for the model.
    Args:
        problems: List of problems.
        tokenizer: Tokenizer.
        max_length: Maximum sequence length.
    Returns:
        Dictionary of encoded problems.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded = tokenizer(
        problems,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded

def get_dataset(dataset_name="math", num_samples=10, tokenizer=None):
    """
    Meta function to load a dataset.
    Args:
        dataset_name: Name of the dataset.
        num_samples: Number of samples to load.
        tokenizer: Tokenizer.
    Returns:
        List of problems.
    """
    if dataset_name == "gsm8k":
        return load_gsm8k_problems(num_samples)
    elif dataset_name == "mmlu_logic":
        return load_mmlu_problems("logical_fallacies", num_samples)
    elif dataset_name == "mmlu_history":
        return load_mmlu_problems("high_school_european_history", num_samples)
    elif dataset_name == "code":
        return load_humaneval_problems(num_samples)
    elif dataset_name == "medqa":
        return load_medqa_problems(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}") 