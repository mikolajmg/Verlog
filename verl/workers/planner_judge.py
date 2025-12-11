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
        """
        Ta metoda zostanie wywołana przez trainer.
        Wykonujemy tu 'Sanity Check' - próbne generowanie.
        """
        print("[UnifiedBrain] Sanity Check...")
        
        # 1. Definiujemy prosty prompt testowy
        test_prompt = "Hello, are you ready to work? Answer with one word."
        
        
        params = SamplingParams(temperature=0.0, max_tokens=10)
        
        try:
            # 3. Generujemy
            outputs = self.llm.generate([test_prompt], params)
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
                last_msg['content'] += f"\n\n{PLAN_INSTRUCTION}"
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
            max_tokens=self.config.support_model.planner.max_plan_length,
            stop=["Observation:", "Current Observation:"] 
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        print(f"[UnifiedBrain] Generating plans for {len(prompts)} prompts.")
        print(f"[UnifiedBrain] Sample prompt: {prompts[0]}")
        print(f"[UnifiedBrain] Sample output: {outputs[0].outputs[0].text.strip()}")
        obs_augmented = []
        for history, plan in zip(observation_prompts, outputs):
            generated_text = plan.outputs[0].text.strip()
            plan_message = {
                "role": "assistant", 
                "content": generated_text
            }
            
            history.append(plan_message)
            obs_augmented.append(history)
    
        return obs_augmented,outputs
        

    def evaluate_reward(self, judge_prompts: list[str]) -> list[float]:
        """Funkcja dla Reward Modelu (Sędzia)."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=100, 
        )
        print(f"[UnifiedBrain] Evaluating rewards for {len(judge_prompts)} prompts.")
        

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
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
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