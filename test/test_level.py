# test_all_levels.py

from game.env.game_environment import game_environment

for level in range(1, 6):
    env = game_environment(level=level, obs_type="spatiotemporal")
    obs, _ = env.reset(seed=42)
    static_count = len(env.static_food)
    dynamic_count = len(env.dynamic_food)
    print(f"Level {level}: {static_count} static, {dynamic_count} dynamic")
    
    # Run 100 random steps
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"  -> After 100 steps: reward={total_reward:.2f}, snake_len={len(env.snake)}")
    env.close()