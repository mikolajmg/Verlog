NAIVE_INSTRUCTION = """
Based on your current plan and situation, select the next action to take.

Output exactly this format:
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip()

PLAN_INSTRUCTION = """
Review your previous observations and current situation, then create a focused plan for the next steps.
Your plan should identify the immediate goal and approach.

Output in exactly this format:
<plan>Your plan here</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip()

DID_PLAN_COMPLETE_QUESTION = """
Based on the most recent observation and the current plan, has the plan been fully completed?
Answer strictly with a single token: YES or NO.
""".strip()

DIVERSE_PLAN_INSTRUCTIONS = [
    """
Identify the immediate next subgoal required to progress towards the overall task completion.
Outline your plan to achieve this specific subgoal, including any necessary reasoning.

Output your plan strictly in the following format:
<plan>YOUR_PLAN_FOR_SUBGOAL</plan>
<action>ActionName</action>

Replace YOUR_PLAN_FOR_SUBGOAL with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that initiates this plan.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Consider the overall objective.
What is the most crucial intermediate milestone to achieve next?
Explain why reaching this milestone is important for the overall task, and outline the steps you'll take to get there.

Output your reasoning and plan strictly in the following format:
<plan>YOUR_REASONING_AND_SUBGOAL_PLAN</plan>
<action>ActionName</action>

Replace YOUR_REASONING_AND_SUBGOAL_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list to start working towards this milestone.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Detail the specific sequence of actions you intend to take over the next few steps.
Explain the purpose of this sequence in relation to the current situation.

Output your detailed short-term plan strictly in the following format:
<plan>YOUR_DETAILED_SHORT_TERM_PLAN</plan>
<action>ActionName</action>

Replace YOUR_DETAILED_SHORT_TERM_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list, which should be the first step in your detailed plan.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Think step-by-step for the immediate future.
What actions are needed right now and why?
Describe the logic connecting these immediate actions to the next phase of the task.

Output your step-by-step thinking and plan strictly in the following format:
<plan>YOUR_STEP_BY_STEP_LOGIC_AND_PLAN</plan>
<action>ActionName</action>

Replace YOUR_STEP_BY_STEP_LOGIC_AND_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that represents the very next concrete step.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Analyze the current state and the final goal.
Formulate a plan that bridges the gap, focusing on the most logical next phase of work.
Explain how this phase contributes to the overall objective.

Output your analysis and plan strictly in the following format:
<plan>YOUR_BRIDGING_PLAN_AND_REASONING</plan>
<action>ActionName</action>

Replace YOUR_BRIDGING_PLAN_AND_REASONING with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list to begin executing this phase.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Re-evaluate the overall strategy.
Outline your current high-level plan or strategic direction for completing the task from this point forward, focusing on the major phases ahead.

Output your strategic plan strictly in the following format:
<plan>YOUR_HIGH_LEVEL_STRATEGIC_PLAN</plan>
<action>ActionName</action>

Replace YOUR_HIGH_LEVEL_STRATEGIC_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that aligns with the first step of this strategy.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Propose a plan for the next stage of the task.
Critically, justify *why* this sequence of steps (or this approach) is the most sensible course of action right now.

Output your plan and justification strictly in the following format:
<plan>YOUR_PLAN_WITH_JUSTIFICATION</plan>
<action>ActionName</action>

Replace YOUR_PLAN_WITH_JUSTIFICATION with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that initiates your justified plan.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Verbalize your thought process for deciding what to do next.
Explain your reasoning, considering the current situation and the ultimate goal, and then state your resulting plan for the near term.

Output your reasoning process and plan strictly in the following format:
<plan>YOUR_REASONING_PROCESS_AND_PLAN</plan>
<action>ActionName</action>

Replace YOUR_REASONING_PROCESS_AND_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list based on your reasoning.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Briefly consider possible approaches for the next steps.
State the approach you choose to take and why it seems preferable to alternatives right now.
Outline the plan based on this chosen approach.

Output your chosen approach, rationale, and plan strictly in the following format:
<plan>YOUR_CHOSEN_APPROACH_RATIONALE_AND_PLAN</plan>
<action>ActionName</action>

Replace YOUR_CHOSEN_APPROACH_RATIONALE_AND_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that corresponds to your chosen approach.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Devise a plan to make progress efficiently.
What is the most direct path to achieving the next significant step or subgoal?
Outline this efficient path.

Output your efficiency-focused plan strictly in the following format:
<plan>YOUR_EFFICIENT_PLAN</plan>
<action>ActionName</action>

Replace YOUR_EFFICIENT_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that represents the first step on this path.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Is there critical information missing?
If so, formulate a plan focused on gathering the necessary information or resolving key uncertainties before proceeding with the main task execution.
If not, state your plan for the next execution steps.

Output your information-gathering or execution plan strictly in the following format:
<plan>YOUR_INFORMATION_OR_EXECUTION_PLAN</plan>
<action>ActionName</action>

Replace YOUR_INFORMATION_OR_EXECUTION_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list relevant to this plan.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Describe the logical sequence of operations you intend to perform next.
Explain the dependency: why does step B follow step A? Focus on the immediate sequence.

Output your logical sequence and rationale strictly in the following format:
<plan>YOUR_LOGICAL_SEQUENCE_PLAN</plan>
<action>ActionName</action>

Replace YOUR_LOGICAL_SEQUENCE_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list representing the first operation in your sequence.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Define your immediate goal for the next few actions.
Construct a plan specifically aimed at achieving this immediate goal.

Output your immediate goal and plan strictly in the following format:
<plan>YOUR_IMMEDIATE_GOAL_AND_PLAN</plan>
<action>ActionName</action>

Replace YOUR_IMMEDIATE_GOAL_AND_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that starts this plan.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Considering the available actions and the task objective, formulate a practical plan for the next steps.
What needs to be done now to make steady progress?

Output your practical plan strictly in the following format:
<plan>YOUR_PRACTICAL_PROGRESS_PLAN</plan>
<action>ActionName</action>

Replace YOUR_PRACTICAL_PROGRESS_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list to implement the first step.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
State your intention for the next phase of action.
What do you aim to accomplish in the near future, and what's the general approach?

Output your statement of intent and approach strictly in the following format:
<plan>YOUR_INTENTION_AND_APPROACH</plan>
<action>ActionName</action>

Replace YOUR_INTENTION_AND_APPROACH with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list that reflects this intention.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
    """
Outline your plan for what to do next.
Keep it focused on the immediate steps required.

Output your plan strictly in the following format:
<plan>YOUR_NEXT_STEPS_PLAN</plan>
<action>ActionName</action>

Replace YOUR_NEXT_STEPS_PLAN with your own plan.
Replace ActionName with exactly ONE action from the allowed actions list to start.

Keep your plan relatively brief, only focusing on important information.
Output no other text.
""".strip(),
]

SHORT_DIVERSE_PLAN_INSTRUCTIONS = [
    """
Identify the immediate next subgoal needed for progress and briefly state how you will achieve it.

Output in exactly this format:
<plan>Your subgoal and approach</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
What is the key milestone to reach next? State why it matters and how you'll achieve it.

Output in exactly this format:
<plan>Your milestone and reasoning</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
List the next steps in brief, focusing on what you'll do and why it's needed.

Output in exactly this format:
<plan>Your step sequence and rationale</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Lay out the immediate steps and their purpose.

Output in exactly this format:
<plan>Your step sequence and purpose</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Note the current state vs. final goal. State the most logical next work phase and why.

Output in exactly this format:
<plan>Your next phase and rationale</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Summarize your current high-level approach for task completion.

Output in exactly this format:
<plan>Your strategic approach</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Propose the next stage and justify the choice briefly.

Output in exactly this format:
<plan>Your next stage and justification</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
State your thought process for the next move and summarize your short-term plan.

Output in exactly this format:
<plan>Your reasoning and short-term plan</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Name the approach you'll take next, why, and your brief plan.

Output in exactly this format:
<plan>Your approach and rationale</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Pick the most direct next step and state your short plan.

Output in exactly this format:
<plan>Your direct approach and plan</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
If information is missing, say what and how you'll get it. If not, briefly state your next step.

Output in exactly this format:
<plan>Your information or execution plan</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Lay out your next operations in minimal detail, noting why B follows A.

Output in exactly this format:
<plan>Your operation sequence and rationale</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
State your immediate goal and a brief plan to reach it.

Output in exactly this format:
<plan>Your immediate goal and approach</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Briefly say what needs doing next for progress.

Output in exactly this format:
<plan>Your progress plan</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
State your near-term aim and briefly how you'll approach it.

Output in exactly this format:
<plan>Your aim and approach</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
    """
Briefly outline your next step.

Output in exactly this format:
<plan>Your next step outline</plan>
<action>ActionName</action>

Replace "ActionName" with exactly one action from the valid actions list.
Output nothing else.
""".strip(),
]

DYNAMIC_PLAN_INSTRUCTION = """
Decide whether you need a fresh plan before acting:
- If you already have a valid plan, skip replanning and output only the next action.
- If you need a new plan, write it in <plan>...</plan> and then output the action.

Always end with exactly one action in this format:
<action>ActionName</action>

Replace ActionName with a single valid action. Output nothing else besides an optional <plan> block and the action.
""".strip()
