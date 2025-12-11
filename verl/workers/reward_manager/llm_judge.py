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

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional, List, Dict, Any
import torch
from transformers import PreTrainedTokenizer
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl import DataProto


class LLMJudgeRewardManager:
    

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = 'data_source',
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        prompt_template: str = "Score the following response from 0 to 1:\nPrompt: {prompt}\nResponse: {response}",
        system_prompt: str = "You are a helpful assistant that scores responses.",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        step_freq: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.model_name = model
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.step_freq = step_freq
        
        
        print(f"[LLM Judge] Loading local model: {model}...")
        self.judge_tokenizer = AutoTokenizer.from_pretrained(model)
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            model, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
            )
        print(f"[LLM Judge] Model loaded.")

    def _query_local(self, prompt: str, response: str) -> float:
        formatted_prompt = self.prompt_template.format(prompt=prompt, response=response)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        text = self.judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.judge_tokenizer(text, return_tensors="pt").to(self.judge_model.device)
        
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs, 
                max_new_tokens=self.max_tokens, 
                temperature=self.temperature,
                do_sample=self.temperature > 0
            )
            
        content = self.judge_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_score(content)

    def _parse_score(self, content: str) -> float:
        try:
            score = float(content.strip())
            return score
        except ValueError:
            import re
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            if matches:
                return float(matches[-1])
            return 0.0

    async def _query_judge(self, prompt: str, response: str) -> float:
        """
        Query the LLM Judge and parse the score.
        """
        if self.mode == 'local':
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._query_local, prompt, response)

        formatted_prompt = self.prompt_template.format(prompt=prompt, response=response)
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            content = completion.choices[0].message.content
            return self._parse_score(content)
                
        except Exception as e:
            print(f"Error querying judge: {e}")
            return 0.0

    async def parallel_compute_score_async(self, prompts: List[str], responses: List[str]) -> List[float]:
        tasks = [
            self._query_judge(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]
        return await asyncio.gather(*tasks)

    def __call__(self, data: DataProto, return_dict: bool = False):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # Decode inputs
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        
        prompts_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        # Determine which samples to score
        indices_to_score = []
        if 'step_idx' in data.non_tensor_batch:
            step_indices = data.non_tensor_batch['step_idx']
            for i, step in enumerate(step_indices):
                if step % self.step_freq == 0:
                    indices_to_score.append(i)
        else:
            # If no step info, score all (or handle as default)
            indices_to_score = list(range(len(data)))

        # Prepare lists for scoring
        prompts_to_score = [prompts_str[i] for i in indices_to_score]
        responses_to_score = [responses_str[i] for i in indices_to_score]
        
        # Compute scores asynchronously
        scores_map = {} # map index -> score
        if prompts_to_score:
            try:
                scores = asyncio.run(self.parallel_compute_score_async(prompts_to_score, responses_to_score))
                for idx, score in zip(indices_to_score, scores):
                    scores_map[idx] = score
            except Exception as e:
                print(f"Error in batch reward computing: {e}")
                
        # Assign scores to reward tensor
        # Use env reward for non-scored items if available
        env_rewards = data.batch.get('reward', torch.zeros(len(data)))
        
        for i in range(len(data)):
            if i in scores_map:
                score = scores_map[i]
                # Print examination
                if i < self.num_examine:
                    print(f"[LLM Judge] Prompt: {prompts_str[i][:50]}...")
                    print(f"[LLM Judge] Response: {responses_str[i][:50]}...")
                    print(f"[LLM Judge] Score: {score}")
            else:
                score = env_rewards[i].item() if isinstance(env_rewards, torch.Tensor) else env_rewards[i]
                
            reward_tensor[i, valid_response_length[i].item() - 1] = score
            

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
