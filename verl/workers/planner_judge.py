import ray
from vllm import LLM, SamplingParams, TokensPrompt
import re
import time
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent
)

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
        PLAN_INSTRUCTION = f"""
        Review your previous observations and current situation, then create a focused plan what to do next.
        Your plan must identify the immediate goal and what would be the outcome.
        Output in exactly this format and nothing else:
        <plan>
        <Goal>Your immidiate goal</Goal>
        <outcome>Your outcome here</outcome>
        </plan>
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
            max_tokens = 250000,
            
        )
        
        outputs = self.llm.generate(prompts, sampling_params,use_tqdm=False)
        
        obs_augmented = []
        plan_messages = []
        for history, plan in zip(observation_prompts, outputs):
            generated_text = plan.outputs[0].text.strip()
            extracted_plan = extract_plan(generated_text)
            if extracted_plan:
                generated_text = extracted_plan[:self.config.support_model.planner.max_plan_length]
            else:
                print(f"[UnifiedBrain] Warning: No plan found in output: {generated_text}")
                generated_text  =generated_text[-self.config.support_model.planner.max_plan_length:] 
            plan_message = {
                "role": "system", 
                "content": "Here is a plan for your upcoming turns. Please try to follow it:\n\n"+ generated_text
            }
            
            history.append(plan_message)
            obs_augmented.append(history)
            plan_messages.append([plan_message])
    
        return obs_augmented,plan_messages
        

    
    def evaluate_reward(self, judge_payloads: list[dict]) -> list[float]:
        if not judge_payloads:
            return []

        formatted_prompts = [self._construct_judge_prompt(p) for p in judge_payloads]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=25000, 
            stop=["###", "\n\n\n"] 
        )


        outputs = self.llm.generate(formatted_prompts, sampling_params, use_tqdm=False)
        scores = []
        
        for i, output in enumerate(outputs):
            prompt_text = formatted_prompts[i]
            response_text = output.outputs[0].text.strip()
            score = self._parse_score(response_text)
            
            
            
            scores.append(score)
        
        return scores

    def _construct_judge_prompt(self, payload: dict) -> str:
        
        observations = payload.get("observations", [])
        plan = payload.get("plans", "")

        history_text = ""
        if not observations:
            history_text = "No prior observations (Start of task)."
        else:
            for i, obs in enumerate(observations):
                
                clean_obs = str(obs).strip()
                history_text += f"Step {i+1} Observation:\n{clean_obs}\n\n"

        prompt = f"""### Instruction
You are an expert critic of reasoning trajectories. Your goal is to validate if a set of actions is logical and consistent with a created plan.
The plan is a short description of intended actions to achieve a goal and it also has intended outcome after execution.  
Execution history contains observations and actions made by an actor which was given specific prompt that you can see in the first and last observation. You also get to see intermediate actions taken.
### Proposed  Plan
"{plan}"
### Execution History
{history_text}


### Evaluation Criteria
1. **Consistency**: The actions must not contradict plan.
2. ** Relevance**: The actions must contribute towards achieving the plan's goal.

### Task
Evaluate the Proposed set of actions are:
- logical and follow strictly a plan, classify it as **GOOD**.
- hallucinations or nonsensical, classify it as **BAD**.

### Output Format
Provide a short reasoning sentence, followed by the final result enclosed in double brackets on the last line.
Example:
The plan logic is sound and follows the previous observation because agent gathered wood and it got wood in the inventory.
[[GOOD]]

### Response
"""
        return prompt


    def _parse_score(self, text: str) -> float:
        try:
            text = text.upper().strip()
            if "[[GOOD]]" in text: return 1.0
            if "[[BAD]]" in text:  return 0.0
            
            last_line = text.split('\n')[-1]
            if re.search(r'\bGOOD\b', last_line): return 1.0
            if re.search(r'\bBAD\b', last_line):  return 0.0
            
            return 0.0
        except Exception as e:
            print(f"[Judge] Parsing Error: {e}")
            return 0.0