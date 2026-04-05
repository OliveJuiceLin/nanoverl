# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess GSM8K into the JSONL format expected by nanoverl.

# 原始样本可能是:
# {
#   "question": "Janet's ducks lay 16 eggs per day...",
#   "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9... #### 9"
# }

# process_fn 将其转换为:
# {
#   "data_source": "openai/gsm8k",
#   "prompt": [{"role": "user", "content": "Janet's ducks... Let's think step by step..."}],
#   "ability": "math",
#   "reward_model": {"style": "rule", "ground_truth": "9"},
#   "extra_info": {
#     "split": "train",
#     "index": 0,  # with_indices=True 提供的索引
#     "answer": "原始答案",
#     "question": "原始问题"
#   }
# }
"""

import argparse
import os
import re

import datasets

# from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str) -> str:
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset. Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    
    
    # parser.add_argument("--hdfs_dir", default=None, help="if specified, the preprocessed dataset will be copied to the hdfs directory.")
    
    
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        # 当你直接从 Hugging Face 加载 openai/gsm8k 时，它默认使用 "main" 配置。
        # 但当你加载本地保存的 dataset (如你截图中的 arrow 文件) 时，datasets 库默认将配置注册为 "default"。
        # 1. 远端加载时的 "main"： 官方的 openai/gsm8k 数据集实际上包含两个不同的配置（子集）：
        # main：标准的问答对。
        # socratic：包含苏格拉底式推理拆解的问答对。 因此，当你从网上的 Hugging Face Hub 下载时，你需要指定 "main" 才能告诉代码你要下载哪一个子集。
        # 2. 本地加载时的 "default"： 当你把数据保存为本地的 .arrow、.json 或 .parquet 文件后，这些文件其实只是一堆纯粹的数据。它们已经脱离了 Hugging Face Hub 上那种“多子集”的复杂结构。 当你用 load_dataset 读取这个纯本地目录时，datasets 库为了统一内部结构，仍然需要给它分配一个配置名。因为本地目录没有定义复杂的变体，框架会自动分配一个默认的名字——也就是 "default"。
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # This script keeps the prompt in chat-message form so instruct
            # tokenizers with a built-in chat template can render it correctly
            # during rollout, while still fitting the built-in JSONL loader.
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # hdfs_dir = args.hdfs_dir
    # local_save_dir = args.local_dir
    # if local_save_dir is not None:
    #     print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    # else:
    #     local_save_dir = args.local_save_dir
    local_save_dir = args.local_save_dir
    train_dataset.to_json(os.path.join(local_save_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(local_save_dir, "test.jsonl"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)

    #     copy(src=local_save_dir, dst=hdfs_dir)
