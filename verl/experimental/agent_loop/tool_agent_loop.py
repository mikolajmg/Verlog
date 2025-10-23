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
import json
import logging
import os
from typing import Any
from uuid import uuid4
import numpy as np
import copy

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
    
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], env, counter, env_idx) -> AgentLoopOutput:
        
        
        messages, info = env.get_last_obs()

        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            ),
        )
        
        output = []
        # user_turns, assistant_turns = 0, 0
        num_turns = 0
        while True:
            
            metrics = {}
            with simple_timer("generate_sequences", metrics):
                
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, env_idx=env_idx,
                )
            
            # truncate response_ids to response_length
            response_ids = response_ids[: self.response_length]
            response_mask = [1] * len(response_ids)
            
            # TODO: decode actions from the response ids
            actions = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            
            metrics.update(info.get("metrics", {}))
            
            last_prompt_ids = copy.deepcopy(prompt_ids)
            is_full = await counter.is_full.remote()
            if is_full:
                break
            
            messages, reward, terminated, truncated, info = env.step(actions)
            done = np.logical_or(terminated, truncated)
            
            turn_data = AgentLoopOutput(
                prompt_ids=copy.deepcopy(prompt_ids),
                response_ids=copy.deepcopy(response_ids),
                response_mask=copy.deepcopy(response_mask),
                metrics=copy.deepcopy(metrics),
                reward=reward,
                done=done,
                num_turns=num_turns,
                env_idx=env_idx,
            )
            num_turns += 1
            
            # batch_size = await counter.get_batch_size.remote()
            # mini_batch_size = int(batch_size // 32)
            # if num_turns == mini_batch_size + 1:
            #     break
            
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
                ),
            )
            
            is_full = await counter.increment.remote()
            if is_full:
                break # will discard the last turn data
            else:
                output.append(turn_data)
            
        turn_data = AgentLoopOutput(
            prompt_ids=last_prompt_ids,
            response_ids=[output[-1].response_ids[0]] if output else [151645],
            response_mask=[1],
            metrics=dict(),
            reward=0.0,
            done=True,
            num_turns=num_turns,
            env_idx=env_idx,
        )
        output.append(turn_data)
        
        return output

    async def _call_tool(self, tool_call: FunctionCall) -> dict[str, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]

            instance_id = await tool.create()
            tool_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]

        return {
            "role": "tool",
            "content": tool_response,
        }
