import subprocess, json
import numpy as np
from sklearn.linear_model import Ridge
import uuid
from dataclasses import dataclass
from collections import Counter


@dataclass(frozen=True)
class Item:
	id: uuid.UUID
	player_id: uuid.UUID
	importance: float
	subjects: tuple[int, ...]


def subjects_from_items(items: list[Item]) -> set[int]:
	return [subject for item in items if item is not None for subject in item['subjects']]


def monotonic_subjects(history: list[Item]) -> list[int]:
	monotonic_subjects = []

	if len(history) >= 3:
		counter = Counter(subjects_from_items(history[-3:]))
		for x, y in counter.items():
			if y > 2:
				monotonic_subjects.append(x)
	return monotonic_subjects


def preference_score(item: Item, preferences: list[int]):
	return sum(1 - (preferences[s] / len(preferences)) for s in item['subjects']) / len(
		item['subjects']
	)


def was_last_round_pause(history: list[Item]) -> bool:
	return len(history) >= 1 and history[-1] is None


def current_context(history: list[Item]):
	context_subjects: list[int] = []
	for i in range(-1, -3, -1):
		if len(history) >= -i and history[i]:
			for subject in history[i]['subjects']:
				context_subjects.append(subject)
		else:
			break

	return context_subjects


def get_last_n_subjects(history: list[Item], n: int) -> set[int]:
	return set(subject for item in history[-n:] if item is not None for subject in item['subjects'])


def coherence_score(item: Item, context_subjects: list[Item]):
	coherence_score = 0.0
	if not item:
		return coherence_score
	for subject in item['subjects']:
		item_has_current_subject = subject in context_subjects
		if item_has_current_subject:
			coherence_score += 1

	return coherence_score if coherence_score != 0.0 else -1


def compute_bonuses(
	item: Item,
	history: list[Item],
	monotonic_subjects: list[int],
	preferences: list[int],
) -> list[float]:
	def repetition_bonus(item: Item, history: list[Item]) -> float:
		return -1 if item and item in history else 0

	def preference_bonus(item: Item, history: list[Item]) -> float:
		return preference_score(item, preferences) if item and item not in history else 0

	def importance_bonus(item: Item, history: list[Item]) -> float:
		return item.importance if item and item not in history else 0

	def freshness_bonus(item: Item, history: list[Item]) -> float:
		if not item:  # pauses contribute to the freshness score
			return 2
		if not was_last_round_pause(history):
			return 0
		recent_subjects = get_last_n_subjects(history, 6)
		return sum(1 for s in item['subjects'] if s not in recent_subjects) / 2

	def pause_bonus(item: Item) -> float:
		return 1 if item is None else 0

	def coherence_bonus(item: Item, history: list[Item]) -> float:
		if not item or item in history:
			return 0
		context_subjects = set(current_context(history))
		overlap = set(item['subjects']) & context_subjects
		return coherence_score(item, overlap)

	def monotonic_bonus(item: Item, monotonic_subjects: list[int]) -> float:
		if not item:
			return 0
		return sum(-1 for s in item['subjects'] if s in monotonic_subjects)

	# Collect all bonuses
	return [
		freshness_bonus(item, history),
		coherence_bonus(item, history),
		monotonic_bonus(item, monotonic_subjects),
		repetition_bonus(item, history),
		importance_bonus(item),
		preference_bonus(item, history),
		pause_bonus(item),
	]


def run_simulation(length=100, player='p8', player_count=10, seed=None):
	"""Run simulator once and return parsed JSON."""
	cmd = [
		'uv',
		'run',
		'python',
		'main.py',
		'--length',
		str(length),
		'--memory_size',
		str(length // player_count),
		'--player',
		player,
		str(player_count),
	]
	if seed is not None:
		cmd += ['--seed', str(seed)]

	result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

	print(result.stderr)
	print(result.stdout)
	json_str = result.stdout[result.stdout.index('{') :]
	return json.loads(json_str)


# def reconstruct_history_up_to(turn_index: int, turn_impact: list[dict]) -> list:
#     """
#     Returns the list of items contributed in the conversation up to the given turn index.
#     Pauses are represented as None.
#     """
#     history = []
#     for i in range(turn_index):
#         item = turn_impact[i].get("item")
#         history.append(item)  # item is None if it was a pause
#     return history
# import subprocess
# import json
# import numpy as np
# from sklearn.linear_model import Ridge
# import subprocess, json
# import numpy as np
# from sklearn.linear_model import Ridge
# from collections import Counter
# import subprocess, json
# import numpy as np
# from sklearn.neural_network import MLPRegressor


# def build_ranking_dataset(data):
#     """Build dataset from simulator data."""
#     dataset = []

#     for player_score in data["scores"]["player_scores"]:
#         player_id = player_score["id"]
#         preferences = player_score["preferences"]
#         memory_bank = player_score["memory_bank"]

#         for turn_idx, turn in enumerate(data["turn_impact"]):
#             if turn.get("speaker_id") != player_id:
#                 continue

#             history = reconstruct_history_up_to(turn_idx, data["turn_impact"])
#             actual_item = turn.get("item")
#             monotonic = monotonic_subjects(history)

#             # candidate items
#             candidate_items = [item for item in memory_bank if item not in history]
#             if actual_item not in candidate_items:
#                 candidate_items.append(actual_item)

#             features_list = [compute_bonuses(item, history, monotonic, preferences=preferences)
#                              for item in candidate_items]

#             for i, f_i in enumerate(features_list):
#                 for j, f_j in enumerate(features_list):
#                     if i == j:
#                         continue
#                     y = 1 if candidate_items[i] == actual_item else 0
#                     dataset.append((f_i, y))

#     X = np.array([x for x, _ in dataset], dtype=np.float32)
#     y = np.array([y for _, y in dataset], dtype=np.float32)
#     return X, y


# def run_and_train(length=100, player="p8", player_count=10, seeds=range(1, 21), batch_size=5):
#     """Run simulator in batches and train an MLP perceptron."""
#     all_X, all_y = [], []
#     shared_scores = []

#     for batch_start in range(0, len(seeds), batch_size):
#         batch_seeds = seeds[batch_start: batch_start + batch_size]
#         for seed in batch_seeds:
#             data = run_simulation(length=length, player=player, player_count=player_count, seed=seed)
#             shared_scores.append(data["scores"]["shared_score_breakdown"]["total"])
#             print(f'Seed: {seed} Shared Score: {data["scores"]["shared_score_breakdown"]}')
#             X, y = build_ranking_dataset(data)
#             all_X.append(X)
#             all_y.append(y)

#         avg_shared = np.mean(shared_scores)
#         print(f"[Batch {batch_start//batch_size + 1}] Average Shared Score so far: {avg_shared:.2f}")

#     X_all = np.vstack(all_X)
#     y_all = np.concatenate(all_y)

#     perceptron = MLPRegressor(
#         hidden_layer_sizes=(),  # single linear layer
#         activation='identity',
#         solver='adam',
#         learning_rate_init=0.01,
#         max_iter=500
#     )
#     perceptron.fit(X_all, y_all)

#     print("Training complete.")
#     print("Learned weights:", perceptron.coefs_[0].flatten())
#     print("Bias:", perceptron.intercepts_[0][0])

#     return perceptron, avg_shared


# Example usage
# perceptron, avg_score = run_and_train(length=100, seeds=range(1, 100), batch_size=5)
# After training your regression
# np.save("learned_weights.npy", perceptron.coefs_)
# np.save("learned_intercept.npy", perceptron.intercepts_)
i = 0
while True:
  run_simulation(101, seed=i)
  i+=1
