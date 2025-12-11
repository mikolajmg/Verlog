import torch
import ray
from verl import DataProto
from collections import defaultdict

class BridgeRewardManager:
    """
    BridgeRewardManager zastępuje NaiveRewardManager.
    Zamiast liczyć nagrodę lokalnie funkcją pythonową, wysyła dane do UnifiedBrain (Ray Actor).
    Realizuje logikę: Total Reward = Env Reward + (LLM Reward if active else 0).
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        
        self.judge_enabled = kwargs.get('judge_enable', True)
        self.judge_freq = kwargs.get('judge_frequency', 1)
        
        self.step_counter = 0
        self.brain = None
        print(f"[BridgeManager] Initialized. Judge Enabled: {self.judge_enabled}, Freq: {self.judge_freq}")

    def __call__(self, data: DataProto, return_dict=False):
        """
        Główna pętla wywoływana przez RayPPOTrainer.fit()
        """
        self.step_counter += 1
        
        # 1. Przygotowanie tensora wyjściowego (tak samo jak w Naive)
        # Kształt: [batch_size, response_length]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # 2. Pobranie nagrody ze środowiska (Environment Reward)
        # RayPPOTrainer/RolloutWorker zapisał nagrodę z gry w data.batch['reward']
        # Kształt: [batch_size]
        env_rewards = data.batch['reward']
        batch_size = env_rewards.shape[0]
        device = env_rewards.device

        # 3. Obliczenie nagrody LLM (Judge)
        llm_scores = torch.zeros_like(env_rewards) # Domyślnie zera
        
        should_judge = (self.judge_enabled and (self.step_counter % self.judge_freq == 0))

        
        if self.brain is None:
            try:
                self.brain = ray.get_actor("UnifiedBrain")
            except Exception as e:
                print(f"[BridgeManager] Warning: UnifiedBrain not found: {e}")

        if self.brain:
            # Pobieramy pełne input_ids (Prompt + Response)
            input_ids = data.batch['input_ids']
            input_ids_list = input_ids.tolist()

            try:
                
                futures = self.brain.evaluate_reward.remote(input_ids_list)
                scores_list = ray.get(futures)
                
                llm_scores = torch.tensor(scores_list, device=device, dtype=torch.float32)
                print(f"[BridgeManager] LLM Scores computed.")
                print(f"[BridgeManager] LLM Scores Tensor: {llm_scores}")
                # Logowanie kilku przykładów dla pewności
                if self.step_counter % 10 == 0:
                    print(f"[BridgeManager] Sample LLM Scores: {scores_list[:3]}")
                    
            except Exception as e:
                print(f"[BridgeManager] Error calculating LLM reward: {e}")
        
        # 4. Sumowanie: Final Score = Environment + LLM
        # Jeśli Judge był wyłączony (skipped), llm_scores to same zera, więc zostaje Environment.
        total_scores = env_rewards + llm_scores

        # 5. Mapowanie wyniku do tensora sekwencji (Sparse Tensor)
        # NaiveRewardManager umieszcza nagrodę na ostatnim tokenie odpowiedzi.
        
        # Obliczamy długość poprawnej odpowiedzi (maska)
        # Responses to [Batch, Resp_Len]
        print(f"[BridgeManager] Batch keys: ")
        
        
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
            
            # Wpisujemy ZSUMOWANĄ nagrodę w odpowiednie miejsce
            reward_tensor[i, last_token_idx] = total_scores[i]

        # 6. Zwracanie wyniku (format zgodny z Naive)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                # Opcjonalnie możemy zwrócić szczegóły do logowania
                "reward_extra_info": defaultdict(list, {"llm_score": llm_scores.tolist(), "env_score": env_rewards.tolist()}) 
            }
        else:
            return reward_tensor