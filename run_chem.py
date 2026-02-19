import torch
import pandas as pd
import re
import json
import os
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig


# ============== 配置 ==============
MODEL_NAME = "OpenDFM/ChemDFM-v1.5-8B"
TEST_DATA_PATH = "data_clean/test_balanced_1.csv"
RESULTS_DIR = "results"

# ============== Prompt 模板 ==============
SYSTEM_PROMPT = """You are a molecular simulation expert with strong knowledge in reaction prediction. Your task is to predict the most likely reaction product and its duration from a given sequence of molecules observed during a molecular dynamics (MD) simulation.

### Background:
- The simulation involves the reaction between Mo3O9 and S2.
- Each molecule is associated with a time duration (in picoseconds).
- Your goal is to determine the final product molecule(s) and their duration(s).

### Format:
Input: (Molecule1, duration1);(Molecule2, duration2);(Molecule3, duration3)
Output: (MoleculeX, durationX)

### Examples:
Input: (MoS5,3);(MoOS6,10);(MoS5,103)
Output: (MoS7,49)

Input: (MoS4,35);(MoOS5,7);(MoS4,54)
Output: (MoS6,36)

Input: (MoOS4,53);(MoS3,143);(MoS5,235)
Output: (MoS7,74)

Now, identify the most likely resulting product molecule and its corresponding duration. Use the patterns observed in previous examples. Return ONLY the predicted result in the format: (Molecule,Time)
If you are uncertain, still give your best guess based on prior examples. Do NOT say 'I don't know'.
"""


def load_test_data(path: str) -> pd.DataFrame:
    """加载测试数据"""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")
    return df


def extract_valid_formulas(df: pd.DataFrame) -> set:
    """从数据集中提取所有有效分子式"""
    formulas = set()
    pattern = r'\(([A-Za-z0-9]+),\d+\)'

    for col in ['X', 'y']:
        for value in df[col]:
            matches = re.findall(pattern, value)
            formulas.update(matches)

    print(f"Found {len(formulas)} valid formulas: {sorted(formulas)}")
    return formulas


def build_prompt(input_x: str) -> str:
    """构建完整的 prompt"""
    prompt = f"[Round 0]\nHuman: {SYSTEM_PROMPT}\nInput: {input_x}\nOutput:\nAssistant:"
    return prompt


def parse_output(output: str) -> tuple:
    """
    解析模型输出，提取分子式和时间
    返回: (molecule, time) 或 (None, None) 如果解析失败
    """
    # 尝试匹配 (Molecule,Time) 或 (Molecule, Time) 格式
    patterns = [
        r'\(([A-Za-z0-9]+)\s*,\s*(\d+)\)',  # (MoS7,49) 或 (MoS7, 49)
        r'([A-Za-z0-9]+)\s*,\s*(\d+)',       # MoS7,49 或 MoS7, 49
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return match.group(1), match.group(2)

    return None, None


def extract_true_molecule(y: str) -> str:
    """从真实标签中提取分子式"""
    match = re.search(r'\(([A-Za-z0-9]+),\d+\)', y)
    return match.group(1) if match else None


def run_inference(model, tokenizer, generation_config, df: pd.DataFrame) -> list:
    """对测试集进行推理"""
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running inference"):
        input_x = row['X']
        true_y = row['y']
        true_molecule = extract_true_molecule(true_y)

        # 构建 prompt
        prompt = build_prompt(input_x)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 生成
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # 解码
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # 提取 Assistant: 之后的内容
        assistant_output = generated_text[len(prompt):].strip()

        # 解析输出
        pred_molecule, pred_time = parse_output(assistant_output)

        results.append({
            'input': input_x,
            'true_output': true_y,
            'true_molecule': true_molecule,
            'raw_output': assistant_output,
            'pred_molecule': pred_molecule,
            'pred_time': pred_time
        })

    return results


def evaluate(results: list, valid_formulas: set) -> dict:
    """计算评估指标"""
    total = len(results)
    correct = 0
    missing = 0
    parse_failed = 0

    for r in results:
        pred = r['pred_molecule']
        true = r['true_molecule']

        if pred is None:
            parse_failed += 1
            missing += 1  # 解析失败也算 missing
        elif pred not in valid_formulas:
            missing += 1
        elif pred == true:
            correct += 1

    metrics = {
        'total_samples': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'missing': missing,
        'missing_rate': missing / total if total > 0 else 0,
        'parse_failed': parse_failed,
        'parse_failed_rate': parse_failed / total if total > 0 else 0
    }

    return metrics


def save_results(results: list, metrics: dict, output_dir: str):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存预测结果
    predictions_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # 保存评估指标
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    print("=" * 50)
    print("ChemDFM Baseline Test")
    print("=" * 50)

    # 加载模型
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    generation_config = GenerationConfig(
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.9,
        max_new_tokens=128,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id
    )

    # 加载测试数据
    print("\nLoading test data...")
    df = load_test_data(TEST_DATA_PATH)

    # 提取有效分子式
    valid_formulas = extract_valid_formulas(df)

    # 运行推理
    print("\nRunning inference...")
    results = run_inference(model, tokenizer, generation_config, df)

    # 评估
    print("\nEvaluating...")
    metrics = evaluate(results, valid_formulas)

    # 打印结果
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total_samples']})")
    print(f"Missing Rate: {metrics['missing_rate']:.4f} ({metrics['missing']}/{metrics['total_samples']})")
    print(f"Parse Failed: {metrics['parse_failed_rate']:.4f} ({metrics['parse_failed']}/{metrics['total_samples']})")

    # 保存结果
    save_results(results, metrics, RESULTS_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
