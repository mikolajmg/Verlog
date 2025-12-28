import ray
from vllm import LLM, SamplingParams, TokensPrompt
import re



@ray.remote(num_gpus=1)
class SupportedModelWorker:
    def __init__(self, config):
        
        super().__init__()
        self.config = config
        
        
        model_path = config.support_model.model_path
        
        print(f"[UnifiedBrain] Inicjalizacja modelu z ścieżki: {model_path}")


        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # 1 GPU
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            
        )
        print("[UnifiedBrain] Support model ready.")

     
    def init_model(self):
        #check if model is working properly
        print("[UnifiedBrain] Sanity Check...")
        
        
        test_prompt = "Hello, are you ready to work? Answer with one word."
        
        
        params = SamplingParams(temperature=0.0, max_tokens=10)
        
        try:
            
            outputs = self.llm.generate([test_prompt], params,use_tqdm=False)
            generated_text = outputs[0].outputs[0].text.strip()
            
            print("="*60)
            print(f"[UnifiedBrain] Prompt: '{test_prompt}'")
            print(f"[UnifiedBrain] Output: '{generated_text}'")
            print("="*60)
            
        except Exception as e:
            print("="*60)
            print(f"[UnifiedBrain] Error: {e}")
            print("="*60)
            raise e

    def generate_plan(self, observation_prompts: list[str]) -> list[str]:
        def extract_plan(text: str) -> str | None:
            matches = re.findall(r"<plan>(.*?)</plan>", text, re.DOTALL)
            return matches[-1].strip() if matches else None

        COT_START_MARKER = "What will you do next?"
        PLAN_INSTRUCTION = """
Review your previous observations and current situation, then create a focused plan for the next steps.
Your plan should identify the immediate goal and approach.

Output in exactly this format:
<plan>Your plan here</plan>
"""

        modified_chats = []
        for chat in observation_prompts:
            new_chat = chat[:] 
            if new_chat and new_chat[-1]['role'] == 'user':
                last_msg = new_chat[-1].copy()
                last_content = last_msg['content']
                cleaned_content = last_content.split(COT_START_MARKER, 1)[0].strip() # take care of the last instriuction prompt
                last_msg['content'] = cleaned_content + f"\n\n{PLAN_INSTRUCTION}"
                new_chat[-1] = last_msg
            else:
                new_chat.append({'role': 'user', 'content': PLAN_INSTRUCTION})
            
            modified_chats.append(new_chat)

        
        tokenizer = self.llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            for chat in modified_chats]
     #TODO: add params to config
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens = 160000,
            
        )
        
        outputs = self.llm.generate(prompts, sampling_params,use_tqdm=False)
        
        #print(f"[UnifiedBrain] Generating plans for {len(prompts)} prompts.")
        #print(f"[UnifiedBrain] Sample output: {outputs[0].outputs[0].text.strip()}")
        obs_augmented = []
        plan_messages = []
        for history, plan in zip(observation_prompts, outputs):
            generated_text = plan.outputs[0].text.strip()
            extracted_plan = extract_plan(generated_text)
            if extracted_plan:
                generated_text = extracted_plan[:self.config.support_model.planner.max_plan_length]
                #print(f"[UnifiedBrain] Extracted plan: {generated_text}")
            else:
                print(f"[UnifiedBrain] Warning: No plan found in output: {generated_text}")
                generated_text  =generated_text[-self.config.support_model.planner.max_plan_length:] 
            plan_message = {
                "role": "user", 
                "content": "Here is a plan for your upcoming turns. Please try to follow it:\n\n"+ generated_text
            }
            
            history.append(plan_message)
            obs_augmented.append(history)
            plan_messages.append([plan_message])
    
        return obs_augmented,plan_messages
        

    def evaluate_reward(self, judge_prompts: list[str]) -> list[float]:
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=100, 
        )
        
        

        formatted_prompts = []
        if len(judge_prompts) > 0:
            first_item = judge_prompts[0]
            
            # Check if the input is a list of integers (Token IDs)
            if isinstance(first_item, list) and len(first_item) > 0 and isinstance(first_item[0], int):
                print("[UnifiedBrain] Detected Token IDs. Wrapping in TokensPrompt.")
                for p in judge_prompts:
                    formatted_prompts.append(TokensPrompt(prompt_token_ids=p))
            else:
                # Assume it is already a string or correct format
                print(f"[UnifiedBrain] Sample prompt type: {type(first_item)}")
                formatted_prompts = judge_prompts
        else:
            formatted_prompts = []
        
        print(f"[UnifiedBrain] Sample prompt: {formatted_prompts[0]}")
        outputs = self.llm.generate(formatted_prompts, sampling_params,use_tqdm=False)
        
        scores = []
        for output in outputs:
            #print(f"[UnifiedBrain] Raw Judge Output: {output.outputs[0].text.strip()}")
            text_response = output.outputs[0].text.strip()
            score = self._parse_score(text_response)
            scores.append(score)
        return scores

    def _parse_score(self, text: str) -> float:
        try:
            match = re.search(r"[-+]?\d*\.\d+|\d+", text)
            if match:
                val = float(match.group())
                if val > 1.0: val = val / 10.0
                return max(0.0, min(1.0, val))
            
            lower_text = text.lower()
            if "yes" in lower_text or "good" in lower_text: return 1.0
            if "no" in lower_text or "bad" in lower_text: return 0.0
            return 0.0
        except Exception as e:
            print(f"[UnifiedBrain] Error in parsing: '{text}': {e}")
            return 0.0