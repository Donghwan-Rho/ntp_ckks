import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import random
import math
import argparse
import os
import numpy as np
import matplotlib
import re
matplotlib.use("Agg")  # headless 환경에서 파일 저장용
import matplotlib.pyplot as plt

from huggingface_hub import login
token_path = os.path.join(os.path.dirname(__file__), "tokens.txt")
hf_token = None
if os.path.exists(token_path):
    with open(token_path, "r", encoding="utf-8") as f:
        hf_token = f.read().strip().strip('"').strip("'")
if hf_token:
    login(token=hf_token)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--max_length", default=1500, type=int)
parser.add_argument("--model_dir", default='wikitext-103-v1/CEloss-with-EFFalse_maxlength4096_lr5e-05_grad-acc32_steps100_lambda1_cos_sim_seed', type=str) # seed 제외
# parser.add_argument("--model_dir", default='wikitext-103-v1/CEloss-with-EFFalse_maxlength4096_lr5e-05_grad-acc32_steps100_lambda1_cos_sim_seed', type=str) # seed 제외
parser.add_argument("--norm", default='cos_sim', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--text_seed", default=42, type=int)
parser.add_argument("--mode", default='greedy', type=str, choices=['greedy', 'top_p'], help='Sampling mode: greedy or top_p')
parser.add_argument("--top_p", default=0.9, type=float, help='Top-p value for nucleus sampling (used when mode=top_p)')
args = parser.parse_args()

def _sanitize_dirname(name: str) -> str:
    """
    prompt를 폴더명으로 쓰기 위해 최소한의 sanitize.
    (슬래시/역슬래시/콜론 등 파일시스템 위험 문자 제거)
    """
    name = name.strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r'[:*?"<>|]', "_", name)  # Windows 금지 문자들도 방지
    name = re.sub(r"\s+", " ", name)        # 공백 정리
    return name if len(name) > 0 else "EMPTY_PROMPT"

def histogram_and_txt(prompt, num, logits, num_bins=1000, hist_min=-30.0, hist_max=30.0,
                      base_dir="generations", min_bar_height=0, mode='greedy', top_p_value=None):
    """
    prompt별 폴더(generations/{prompt}/{mode})를 만들고,
    히스토그램: histogram_{num}.png
    텍스트:     txt_{num}.txt
    로 저장한다.

    logits: torch.Tensor, shape [1, vocab] 또는 [vocab] 등 (flatten해서 처리)
            **mean을 빼지 않은 raw logits을 넣는다.**
    히스토그램과 std/topk/bin-count는 (logits - mean(logits)) 기준으로 계산한다.
    max(softmax)는 raw logits 기준으로 계산한다.
    """
    # 폴더 준비
    prompt_subdir = _sanitize_dirname(prompt)
    if mode == 'greedy':
        mode_dir = 'greedy'
    else:  # mode == 'top_p'
        mode_dir = f'top_p_p{top_p_value}'
    prompt_dir = os.path.join(base_dir, prompt_subdir, mode_dir)
    os.makedirs(prompt_dir, exist_ok=True)

    # 파일 경로
    png_path = os.path.join(prompt_dir, f"histogram_{num}.png")
    txt_path = os.path.join(prompt_dir, f"txt_{num}.txt")

    # raw logits -> 1D torch
    raw_t = logits.detach().float().reshape(-1)  # [vocab]

    # max(softmax) on raw logits (numerically stable)
    max_softmax = torch.softmax(raw_t, dim=-1).max().item()

    # mean-centering (histogram은 이걸로)
    x_t = raw_t - raw_t.mean()
    x_np = x_t.cpu().numpy()

    # std (mean-centered logits 기준)
    std_val = torch.std(x_t, unbiased=False).item()

    # top-100 logits 값 (mean-centered 기준)
    k = min(100, x_t.numel())
    top_vals, _ = torch.topk(x_t, k=k, largest=True, sorted=True)
    top_vals = top_vals.cpu().numpy()

    # Calculate mean((logits except top k outliers)^p) for k=10 and p=1,3,7,15
    k_outliers = 10
    top_k_outliers, _ = torch.topk(x_t, k=min(k_outliers, x_t.numel()), largest=True, sorted=False)
    # Create mask to exclude top k outliers
    mask = torch.ones_like(x_t, dtype=torch.bool)
    for val in top_k_outliers:
        idx = (x_t == val).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            mask[idx[0]] = False
    
    logits_except_topk = x_t[mask]
    mean_powers = {}
    lehmer_means = {}
    lehmer_exceed_counts = {}
    for p in [1, 3, 7, 15, 31]:
        if len(logits_except_topk) > 0:
            # Mean power: mean((x-topk)^p) without absolute value (EXCLUDING outliers)
            mean_powers[p] = torch.mean(logits_except_topk ** p).item()
        else:
            mean_powers[p] = 0.0
        
        # Lehmer mean: sum(relu(x)^p) / sum(relu(x)^(p-1)) (INCLUDING outliers)
        # Scale down by 10 to avoid overflow, then scale result back up
        relu_logits_all = torch.relu(x_t)
        scaled_relu = relu_logits_all / 10.0  # Scale down to prevent overflow
        if p == 1:
            # Special case: Lehmer mean with p=1 is the arithmetic mean
            lehmer_means[p] = torch.mean(scaled_relu).item() * 10.0  # Scale back
        else:
            numerator = torch.sum(scaled_relu ** p)
            denominator = torch.sum(scaled_relu ** (p - 1))
            if denominator != 0:
                lehmer_means[p] = (numerator / denominator).item() * 10.0  # Scale back
            else:
                lehmer_means[p] = 0.0
        
        # Count how many elements exceed lehmer_mean_p (all elements)
        lehmer_exceed_counts[p] = (x_t > lehmer_means[p]).sum().item()

    # histogram counts (진짜 카운트)
    edges = np.linspace(hist_min, hist_max, num_bins + 1, dtype=np.float64)
    counts, _ = np.histogram(x_np, bins=edges)  # 범위 밖은 자동 제외됨

    # --- 히스토그램 그림 저장 (min_bar_height 적용: 시각화용) ---
    bin_width = edges[1] - edges[0]
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    counts_vis = counts.copy()
    if min_bar_height is not None and min_bar_height > 0:
        mask = counts_vis > 0
        counts_vis[mask] = np.maximum(counts_vis[mask], int(min_bar_height))

    plt.figure(figsize=(16, 6))
    plt.bar(bin_centers, counts_vis, width=bin_width, align="center", alpha=0.7, label='Histogram')
    plt.xlim(hist_min, hist_max)
    
    # Plot vertical lines for mean powers
    colors_mean = ['red', 'orange', 'purple', 'brown', 'blue']
    for i, p in enumerate([1, 3, 7, 15, 31]):
        plt.axvline(x=mean_powers[p], color=colors_mean[i], linestyle=':', linewidth=2, 
                    label=f'mean(x^{p})={mean_powers[p]:.3f}')
    
    # Plot vertical lines for Lehmer means
    colors_lehmer = ['darkred', 'darkorange', 'darkviolet', 'saddlebrown', 'darkblue']
    for i, p in enumerate([1, 3, 7, 15, 31]):
        plt.axvline(x=lehmer_means[p], color=colors_lehmer[i], linestyle='--', linewidth=2,
                    label=f'Lehmer_p{p}={lehmer_means[p]:.3f}')
    
    title_str = (
        f"Logits Histogram (mean-centered) | num={num} | bins={num_bins} | range=[{hist_min},{hist_max}]\n"
        f"std {std_val:.6f} | max(softmax) {max_softmax:.6f}"
    )
    plt.title(title_str, fontsize=10)
    plt.xlabel("logit value (x - mean(x))")
    plt.ylabel("count (visualized)")
    plt.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    # --- txt 저장 (진짜 카운트 기록) ---
    width = hist_max - hist_min
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Number of outliers (k): {k_outliers}\n\n")
        f.write(f"std: {std_val:.10f} | max(softmax): {max_softmax:.10f}\n\n")
        
        f.write("Mean power statistics (excluding top 10 outliers):\n")
        for p in [1, 3, 7, 15, 31]:
            f.write(f"  mean((x-topk)^{p:2d}): {mean_powers[p]:.10f}\n")
        f.write("\n")
        
        f.write("Lehmer mean statistics (including all elements):\n")
        for p in [1, 3, 7, 15, 31]:
            f.write(f"  Lehmer_mean_p{p:2d}: {lehmer_means[p]:.10f}  |  elements exceeding this: {lehmer_exceed_counts[p]}\n")
        f.write("\n")

        for i, v in enumerate(top_vals, start=1):
            f.write(f"top {i:3d}: {v:.10f}\n")

        f.write("\n")
        f.write(f"histogram_bins: num_bins={num_bins}, range=[{hist_min}, {hist_max}]\n")
        f.write("(counts exclude values outside the range)\n\n")

        for i in range(num_bins):
            left_expr  = f"{hist_min:g} + {width:g}*{i:4d}/{num_bins}"
            right_expr = f"{hist_min:g} + {width:g}*{i+1:4d}/{num_bins}"
            f.write(f"[{left_expr}, {right_expr}]: {int(counts[i])}\n")

    return png_path, txt_path

# CRITERIA 문자열 설정
CRITERIA = """I’m going to give you a piece of writing. This text was generated by an LLM using random sampling. Please determine whether or not this text is corrupted. The criteria for being considered corrupted are as follows:

When a specific character is repeated meaninglessly. For example, something like Coooooooooooooool! has meaningful repetition, so it wouldn’t be considered corrupted. However, something like MSMSMSMSMS...—a meaningless sequence of repeated characters—would be considered corrupted.

When the arrangement of words is excessively random to the point where the text is completely unintelligible. Random sampling can result in some randomness in sentences, so a text with a reasonable degree of randomness wouldn’t be considered corrupted. However, if the randomness is excessive to the point where the text becomes utterly unreadable, it would be considered corrupted. However, since the current text was generated to match a specific token count, please disregard any incomplete sentences at the end.

After reading the text, assign a score based on the degree of corruption in the following format:  
**X point(s): {REASON}**

Here is the scoring system:  
4 points: If 80-100% of the text is corrupted.  
3 points: If 60-80% of the text is corrupted.  
2 points: If 40-60% of the text is corrupted.  
1 point: If 20-40% of the text is corrupted.  
0 points: If 0-20% of the text is corrupted.  

**Special Case:** Regardless of the above criteria, if the sequence MS is repeated meaninglessly more than two times, assign **4 points**.

Here is the text I’ll show you:
"""

# Enhanced set_seed function to ensure reproducibility
def set_seed(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU random module
    torch.cuda.manual_seed(seed)  # PyTorch GPU random module
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (multi-GPU) random module
    torch.backends.cudnn.deterministic = True  # Ensures deterministic CuDNN behavior
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility
    
    # Transformer-specific random generators, if any
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass  # Transformers not installed or seed function unavailable
    print(f'Seed: {seed}')

# Set seed for reproducibility
set_seed(args.text_seed)

# Perplexity 계산 함수
def calculate_perplexity(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        perplexity = math.exp(loss)
    return perplexity

cache_dir = os.path.expanduser("~/.cache/huggingface")

# 모델과 토크나이저 불러오기
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
last_token_id = len(tokenizer) - 1
print(f'last_token_id: {last_token_id}')

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir)
model.eval()
# print(f'Model:\n{model}')

embedding_weights = model.model.embed_tokens.weight
print(f'Embedding weights: {embedding_weights.shape}')

# 질문 정의
# prompt = "Please introduce yourself."
# prompt = "What are some beginner tips for learning a new language?"
# prompt = "What are some effective time management strategies?"
# prompt = "What are the key differences between renewable and non-renewable energy?“
# prompt = "How can someone start learning programming?"

prompt_list = [
    "Please introduce yourself.",
    # "Tell me about a time you overcame a challenge.",
    # "Describe the U.S. high school math curriculum."
               ]

folder_dir = f"random_generation/{str(args.max_length)}tokens"
# os.makedirs(folder_dir, exist_ok=True)

for prompt in prompt_list:
    print(f'Prompt: {prompt}')
    if len(prompt) > 30:
        prompt_name = prompt[:30]  # 처음 30글자만 가져오기
    else:
        prompt_name = prompt  # 길이가 30 이하인 경우 전체 사용 
    txt_dir = os.path.join(folder_dir, f"{prompt_name}_seed{args.text_seed}.txt")
    sparsity_dir = os.path.join(folder_dir, f"{prompt_name}_seed{args.text_seed}_sparsity.txt")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    num_generation = 1
    meet_eos = False
    num_token = 0
    while True:
        # break
        MS_num = 0
        meet_eos = False
        num_token = 0
        
        msg = ''
        msg += f'\n{num_generation}-th generation.\n\n'
        msg += f'Prompt: {prompt}\n\n'
        msg += f'Generation length: {args.max_length}\n\n'
        if args.mode == 'greedy':
            msg += f'Sampling mode: Greedy decoding (argmax)'
        else:
            msg += f'Sampling mode: Top-p sampling with p={args.top_p}'
        msg += '\n-----------------\n\n'
        
        no_adjustment_text_list = []
        no_adjustment_next_token_list = []
        sparsity_scores = []
        # Top-p sampling with p=1 (without weighting)
        generated_text_no_adjustment = input_ids
        for num in range(args.max_length):    
            outputs = model(generated_text_no_adjustment)
            # print(f'generated_text_no_adjustment: {generated_text_no_adjustment.shape}')
            # print(f'outputs: {outputs.logits.shape}')     
            
            next_token_logits = outputs.logits[:, -1, :]
            mean_centered_token_logits = next_token_logits - next_token_logits.mean(dim=-1, keepdim=True)
            print(f'num: {num:4d} Mean-centered next_token_logits: max {mean_centered_token_logits.max().item()}, min {mean_centered_token_logits.min().item()}')
            histogram_and_txt(prompt, num, next_token_logits, min_bar_height=11, mode=args.mode, top_p_value=args.top_p)
            
            probabilities = torch.softmax(next_token_logits, dim=-1)
            sparse_count = (probabilities != 0).sum().item()
            sparsity_ratio = sparse_count / probabilities.numel()
            sparsity_scores.append(sparsity_ratio)
            
            # Sample next token based on mode
            if args.mode == 'greedy':
                # Greedy decoding (argmax)
                next_token_id = probabilities.argmax(dim=-1).view(-1)
            else:  # args.mode == 'top_p'
                # Top-p (nucleus) sampling
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > args.top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Set probabilities to zero for removed indices
                probs_copy = probabilities.clone()
                probs_copy[0, sorted_indices[0, sorted_indices_to_remove[0]]] = 0
                probs_copy = probs_copy / probs_copy.sum(dim=-1, keepdim=True)
                
                # Sample from the filtered distribution
                next_token_id = torch.multinomial(probs_copy, num_samples=1).view(-1)
            
            no_adjustment_text_list.append(tokenizer.decode([next_token_id.item()]))
            # print(f'gen text: {tokenizer.decode([next_token_id.item()])}')
            no_adjustment_next_token_list.append(next_token_id.item())
            
            generated_text_no_adjustment = torch.cat([generated_text_no_adjustment, next_token_id.unsqueeze(-1)], dim=-1)
            
            if tokenizer.decode([next_token_id.item()], skip_special_tokens=True) in ['MS', ' MS']:
                MS_num += 1

            next_token = tokenizer.decode(next_token_id)
            num_token += 1
            if next_token == tokenizer.eos_token:
                # print(f'Met EOS token. Do not write this answer.\n')
                meet_eos = True
                break

        # msg += f'\nNumber of generated tokens: {num_token}\n'
        # # 생성된 답안 출력 및 Perplexity 계산
        # if meet_eos == False:
        #     with open(sparsity_dir, 'a', encoding='utf-8') as f:
        #         if MS_num < 3:
        #             f.write(f'{np.mean(sparsity_scores):.10f}\n')
        #         else:
        #             f.write('XXX\n')
        #     generated_text_no_adjustment_str = tokenizer.decode(generated_text_no_adjustment[0], skip_special_tokens=True)
        #     # print(colored("\nGenerated response without adjustment:", 'yellow'))
        #     # print(colored(generated_text_no_adjustment_str, 'yellow'))
        #     perplexity = calculate_perplexity(model, tokenizer, generated_text_no_adjustment_str, device)
        #     # print(colored(f"Perplexity: {perplexity}", 'yellow'))

        #     msg += '\n-----------------\n\n'
        #     msg += f'Generated response without adjustment:\n\n{generated_text_no_adjustment_str}\n\n'
        #     msg += f'Perplexity: {perplexity:.4f}\n\n'
        #     msg += f"The number of 'MS' tokens: {MS_num}\n"
        #     msg += '\n----------------------------------------\n\n'
        #     with open(txt_dir, 'a', encoding='utf-8') as f:
        #         f.write(msg)
        #     num_generation += 1
        
        # if num_generation > 100:
        #     print(f'Directory:\n{txt_dir}')
        #     break
    
    # common_path = folder_dir
    # file_name = f"{prompt_name}_seed{args.text_seed}.txt"
    # process_file(common_path=common_path, file_name=file_name)
    # process_response(common_path=common_path, file_name=file_name)
    # output_file_path = os.path.join(common_path, f'{file_name}_corruption_score.txt')
    # for i in range(1, 101):
    #     # if i != 46:
    #     #     continue
    #     # continue
    #     response_i_path = os.path.join(common_path, f'{file_name}_responses', f'response_{i}.txt')
        
    #     # 파일이 존재한다면 읽어서 GPT-4에 전달
    #     if os.path.exists(response_i_path):
    #         # print(f'Processing the {i}-th response...')
    #         with open(response_i_path, 'r', encoding='utf-8') as f:
    #             response = f.read()
            
    #         # GPT-4에게 질문 & 답변
    #         gpt4_response = ask_gpt4(CRITERIA, response)
            
    #         # 결과 저장
    #         with open(output_file_path, 'a', encoding='utf-8') as f:
    #             f.write(f"{i}-th score\n\n")
    #             f.write(f"{CRITERIA}\n\n")
    #             f.write(f"{response}\n")
    #             f.write("GPT-4 Response:\n\n")
    #             f.write(f"{gpt4_response}\n")
    #             f.write("============================================\n")
                
    # # Example usage
    # extract_scores(output_file_path, model_name='GPT-4')
    # print(f'Directory:\n{output_file_path}')
