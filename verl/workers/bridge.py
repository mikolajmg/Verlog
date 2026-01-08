import torch
import ray
from verl import DataProto
from collections import defaultdict

class BridgeRewardManager:
    """ Bridge Reward Manager integrating LLM-based judgment into reward calculation. """
    def __init__(self, tokenizer, num_examine, compute_score=None ,reward_fn_key='data_source',config=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.judge_freq = num_examine
        self.reward_fn_key = reward_fn_key
        self.kwargs = kwargs
        self.judge_enabled = config.support_model.judge.enable
        #self.judge_freq = kwargs.get('reward_kwargs', {}).get('judge_frequency', 5)
        #self.judge_freq = self.judge_freq['judge_frequency']
        self.n_rollouts = config.reward_model.n_rollouts

        self.step_counter = 0
        self.brain = None

        self.config = kwargs.get('config', None)
        
    def __call__(self, data: DataProto, return_dict=False):
        print(f"[BridgeManager] kwargs received: {self.kwargs}")
        env_rewards = data.batch['reward']
        
        dones = data.batch['done'].to(torch.bool) 
        
        batch_size = env_rewards.shape[0]
        n_envs = self.n_rollouts
        num_steps = batch_size // n_envs
        device = env_rewards.device
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        llm_scores_flat = torch.zeros(batch_size, device=device)
        

        
        

        
        if self.brain is None:
            try:
                self.brain = ray.get_actor("UnifiedBrain")
            except Exception as e:
                print(f"[BridgeManager] Warning: UnifiedBrain not found: {e}")

        should_judge = (self.judge_enabled and (self.step_counter % self.judge_freq == 0))
        if self.brain and should_judge:
            
            input_ids = data.batch['input_ids']
            responses = data.batch['responses']
            plans = data.batch['plan']
            dones = data.batch['done']
            
            plans_decoded = self.tokenizer.batch_decode(plans, skip_special_tokens=True)
            inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            

            

            batch_indices = []
            payloads = []
            for env_i in range(n_envs):
                
                
                observations=[]
                current_plan=set()
                diff_plan=False
                turns_to_judge = self.judge_freq
                steps_indices = [step * n_envs + env_i for step in range(num_steps)]
                for step in steps_indices:
                    # print(f"[BridgeManager] Decoded plan sample: {plans_decoded[step]}")
                    # print(f"[BridgeManager] Decoded obs sample: {inputs[step]}")
                    
                    if plans_decoded[step] not in current_plan and len(current_plan)>0:
                        
                        diff_plan = True

                    is_done = dones[step]
                    is_limit = (turns_to_judge == 1)
                    is_last = (step == len(steps_indices) - 1)
                    # we ensure we get full observations for the last step and first step
                    
                    if (is_done or is_limit or is_last or turns_to_judge == self.judge_freq) and not diff_plan:
                        
                        observations.append(inputs[step])
                        current_plan.update({plans_decoded[step]})
                    elif not diff_plan:
                        
                        observations.append(responses[step])
                        current_plan.update({plans_decoded[step]})

                    
                    turns_to_judge -= 1
                    if is_done or is_limit or is_last or diff_plan:
                        assert len(set(current_plan))==1,  f"all plans should be the same in this but are:  {observations,current_plan}"
                        payload = {
                            "observations": observations,
                            "plans": set(current_plan).pop(),
                        }
                        # history_text = " \n\n\n___________________________________________\n NEXT OBSERVATION:".join(observations)
                        # print(f"""Plan to evaluate:{plans_decoded[step]} 
                        
                        # for observations  
                        
                        # {history_text}
                        
                        # """)
                        # history_plan = " \n\n\n___________________________________________\n NEXT PLAN:".join(current_plan)
                        # print(f"[BridgeManager] The plans that are being sent to judge: {history_plan}")

                        turns_to_judge=self.judge_freq
                        payloads.append(payload)
                        batch_indices.append(step)
                        current_plan=set()
                        observations=[]
                        diff_plan=False
                        if diff_plan:
                            observations.append(inputs[step])
                            current_plan.update({plans_decoded[step]})
                            diff_plan=False
                    
                        
            

            try:
                
                futures = self.brain.evaluate_reward.remote(payloads)
                scores_list = ray.get(futures)
                print(f"[BridgeManager] LLM Scores received: {mean(scores_list)}")
                for idx, score in zip(batch_indices, scores_list):
                    llm_scores_flat[idx] = score
                #llm_scores = torch.tensor(scores_list, device=device, dtype=torch.float32)
                # print(f"[BridgeManager] LLM Scores computed.")
                # print(f"[BridgeManager] LLM Scores Tensor: {llm_scores}")
                # Logowanie kilku przykładów dla pewności
                if self.step_counter % 10 == 0:
                    print(f"[BridgeManager] Sample LLM Scores: {scores_list[:3]}")
                    
            except Exception as e:
                print(f"[BridgeManager] Error calculating LLM reward: {e}")
        

        total_scores = env_rewards + llm_scores_flat

        
        
        
        responses = data.batch['responses']
        prompt_length = data.batch['attention_mask'].shape[-1] - data.batch['responses'].shape[-1]
        
        # Maska uwagi dla całej sekwencji [Batch, Prompt+Resp]
        attention_mask = data.batch['attention_mask']
        
        # Wyciągamy maskę tylko dla części odpowiedzi
        response_mask = attention_mask[:, prompt_length:]
        
        # Sumujemy maskę, żeby wiedzieć gdzie jest koniec zdania
        valid_response_lengths = response_mask.sum(dim=1)

        for i in range(batch_size):
            # Indeks ostatniego tokenu
            last_token_idx = int(valid_response_lengths[i].item()) - 1
            if last_token_idx < 0: 
                last_token_idx = 0
            
            reward_tensor[i, last_token_idx] = total_scores[i]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": defaultdict(list, {"llm_score": llm_scores_flat.tolist(), "env_score": env_rewards.tolist()}) 
            }
        else:
            return reward_tensor