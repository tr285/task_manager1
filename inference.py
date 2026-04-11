import os
from openai import OpenAI
from environment import MiniGameEnv
from grader import grade_easy, grade_medium, grade_hard
from models import Action

def run_agent(difficulty: str, seed: int = 42):
    print(f"[START] task={difficulty}", flush=True)
    env = MiniGameEnv(difficulty=difficulty)
    obs = env.reset(seed=seed)
    
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )
    
    done = False
    
    while not done:
        agent_x, agent_y = obs.agent_position
        goal_x, goal_y = obs.goal_position
        
        if agent_x == goal_x and agent_y == goal_y:
            break
            
        # Ask LLM for next move
        prompt = f"Agent is at ({agent_x}, {agent_y}). Goal is at ({goal_x}, {goal_y}). What direction should the agent move? Respond only with one of: left, right, up, down."
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            move = response.choices[0].message.content.strip().lower()
            if "right" in move: move = "right"
            elif "down" in move: move = "down"
            elif "left" in move: move = "left"
            elif "up" in move: move = "up"
            else:
                move = "right" if agent_x < goal_x else "down"
        except Exception as e:
            print(f"LLM call failed: {e}", flush=True)
            move = "right" if agent_x < goal_x else "down"
            
        action = Action(direction=move)
            
        obs, reward, done, _ = env.step(action)
        print(f"[STEP] step={env.steps_taken} reward={reward}", flush=True)
        
    return env

if __name__ == "__main__":
    print("Running baseline agent evaluation...", flush=True)
    
    # Easy
    env_easy = run_agent("easy")
    score_easy = grade_easy(env_easy)
    print(f"[END] task=easy score={score_easy:.2f} steps={env_easy.steps_taken}", flush=True)
    
    # Medium
    env_med = run_agent("medium")
    score_med = grade_medium(env_med)
    print(f"[END] task=medium score={score_med:.2f} steps={env_med.steps_taken}", flush=True)
    
    # Hard
    env_hard = run_agent("hard")
    score_hard = grade_hard(env_hard)
    print(f"[END] task=hard score={score_hard:.2f} steps={env_hard.steps_taken}", flush=True)