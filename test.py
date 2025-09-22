import subprocess, json, numpy as np, optuna
import os
from sklearn.linear_model import LinearRegression
import pickle


# ----------------------------
# Simulation runner
# ----------------------------
def run_simulation(length=100, player='p8', memory_size=None, seed=None, env=None):
    if memory_size is None:
        memory_size = length // 10  # fallback

    cmd = [
        'uv',
        'run',
        'python',
        'main.py',
        '--length',
        str(length),
        '--memory_size',
        str(memory_size),
        '--player',
        player,
        '10',  # fixed player count
    ]
    if seed is not None:
        cmd += ['--seed', str(seed)]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.', env=env)
    # print(result.stderr)
    data = json.loads(result.stdout[result.stdout.index('{'):])
    return data


# ----------------------------
# Evaluate given v vector
# ----------------------------
def evaluate_v(v, memory_size=10, conv_length=100, trials=20):
    scores = []
    for i in range(trials):
        env = dict(os.environ)
        env['PLAYER8_V'] = json.dumps(v)
        sim = run_simulation(
            length=conv_length,
            player='p8',
            memory_size=memory_size,
            seed=i,
            env=env,
        )
        score = sim['scores']['shared_score_breakdown']['total']
        scores.append(score)
        print(f"  Trial {i}: score={score}")  # <--- print each trial score
    avg = np.mean(scores)
    print(f"Avg score for v={v}, memory_size={memory_size}, conv_length={conv_length} -> {avg}")
    return avg


# ----------------------------
# Optuna objective
# ----------------------------
def objective(trial):
    candidate_v = [trial.suggest_uniform(f'v{i}', -5, 5) for i in range(6)]
    memory_size = trial.suggest_int("memory_size", 1, 100)
    conv_length = trial.suggest_int("conv_length", 10, 1000)

    # Enforce max constraint
    max_conv_length = memory_size * 10  # since player count = 10
    if conv_length > max_conv_length:
        conv_length = max_conv_length

    avg_score = evaluate_v(
        candidate_v,
        memory_size=memory_size,
        conv_length=conv_length,
        trials=200,
    )

    print(
        f"[Optuna] Params: v={candidate_v}, "
        f"memory_size={memory_size}, conv_length={conv_length} "
        f"(max allowed={max_conv_length}), score={avg_score}"
    )
    return avg_score

# ----------------------------
# Run optimization indefinitely
# ----------------------------
if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///optuna_player8.db',
        study_name='player8_v_mem_conv_search',
        load_if_exists=True,
    )

    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n=== Optimization iteration {iteration} ===")
            # Run a small batch of trials per iteration
            study.optimize(objective, n_trials=5)

            # Print current best
            print(f"Best params so far: {study.best_params}")
            print(f"Best value so far: {study.best_value}")

            # Optional: update regression model each iteration
            X, Y = [], []
            for t in study.trials:
                if t.value is None:
                    continue
                memory_size = t.params["memory_size"]
                conv_length = t.params["conv_length"]
                v = [t.params[f'v{i}'] for i in range(6)]
                X.append([memory_size, conv_length])
                Y.append(v)
            if X:
                model = LinearRegression()
                model.fit(X, Y)
                with open("player8_v_model.pkl", "wb") as f:
                    pickle.dump(model, f)

                default_v = model.predict([[10, 100]])[0].tolist()
                print("Updated default v:", default_v)

    except KeyboardInterrupt:
        print("\nOptimization stopped manually.")
        print(f"Final best params: {study.best_params}")
        print(f"Final best value: {study.best_value}")
