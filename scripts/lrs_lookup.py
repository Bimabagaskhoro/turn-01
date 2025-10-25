import json 
import os 
import hashlib
current_dir = os.path.dirname(os.path.abspath(__file__))

# lr_amplify_rate = 1.05

with open(os.path.join(current_dir, "lrs/dpo.json"), "r") as f:
    dpo_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo.json"), "r") as f:
    grpo_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/instruct.json"), "r") as f:
    instruct_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo_python.json"), "r") as f:
    grpo_python_lrs = json.load(f)


def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 


def get_dpo_lr(model: str, lr_amplify_rate: float):
    hashed_model = hash_model(model)
    for lr in dpo_lrs:
        if lr["h"] == hashed_model:
            val = lr["lr"] * lr_amplify_rate
            print("--------------------------------")
            print("--------------------------------")
            print(
                "=" * 80 + "\n"
                f"[DPO]\n"
                f"model name : {model}\n"
                f"hash_model: {hashed_model}\n"
                f"learning_rate: {val}\n"
                + "=" * 80
            )
            return val
    return None


def get_grpo_lr(model: str, lr_amplify_rate: float):
    hashed_model = hash_model(model)
    for lr in grpo_lrs:
        if lr["h"] == hashed_model:
            val = lr["lr"] * lr_amplify_rate
            print("--------------------------------")
            print("--------------------------------")
            print(
                "=" * 80 + "\n"
                f"[GRPO]\n"
                f"model name : {model}\n"
                f"hash_model: {hashed_model}\n"
                f"learning_rate: {val}\n"
                + "=" * 80
            )
            return val
    return None

def get_instruct_lr(model: str, lr_amplify_rate: float):
    hashed_model = hash_model(model)
    for lr in instruct_lrs:
        if lr["h"] == hashed_model:
            val = lr["lr"] * lr_amplify_rate
            print("--------------------------------")
            print("--------------------------------")
            print(
                "=" * 80 + "\n"
                f"[Instruct]\n"
                f"model name : {model}\n"
                f"hash_model: {hashed_model}\n"
                f"learning_rate: {val}\n"
                + "=" * 80
            )
            return val
    return None

def get_grpo_python_lr(model: str, lr_amplify_rate: float):
    hashed_model = hash_model(model)
    for lr in grpo_python_lrs:
        if lr["h"] == hashed_model:
            val = lr["lr"] * lr_amplify_rate
            print("--------------------------------")
            print("--------------------------------")
            print(
                "=" * 80 + "\n"
                f"[Instruct]\n"
                f"model name : {model}\n"
                f"hash_model: {hashed_model}\n"
                f"learning_rate: {val}\n"
                + "=" * 80
            )
            return val
    return None
