import argparse
from vllm import LLM, SamplingParams


def main(args):
    # 初始化LLM,指定模型和张量并行大小
    llm = LLM(
        model="meta-llama/Llama-3.1-70b-instruct", tensor_parallel_size=args.gpu_count
    )

    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

    # 准备提示词
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot learning to love.",
        "What are the main causes of climate change?",
    ]

    # 执行批量推理
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_count", type=int, default=2, help="Number of GPUs to use"
    )
    args = parser.parse_args()
    main(args)

# python run_llama_on_gpus.py --gpu_count 4
