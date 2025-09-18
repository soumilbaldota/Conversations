from collections import Counter
import numpy as np
from models.player import GameContext, Item, Player, PlayerSnapshot
from players.player_8.player_base import Player8Base


class Player8(Player8Base):
	def compute_bonuses(
		self,
		item: Item,
		history: list[Item],
		monotonic_subjects: list[int],
	) -> list[float]:
		def repetition_bonus(item: Item, history: list[Item]) -> float:
			return -1 if item and item in history else 0

		def preference_bonus(item: Item, history: list[Item]) -> float:
			return self.preference_score(item) if item and item not in history else 0

		def importance_bonus(item: Item, history: list[Item]) -> float:
			return item.importance if item and item not in history else 0

		def freshness_bonus(item: Item, history: list[Item]) -> float:
			if not item:  # pauses contribute to the freshness score
				return 2
			if not self.was_last_round_pause(history):
				return 0
			recent_subjects = self.get_last_n_subjects(history, 6)
			return sum(1 for s in item.subjects if s not in recent_subjects) / 2

		def pause_bonus(item: Item) -> float:
			return 1 if item is None else 0

		def coherence_bonus(item: Item, history: list[Item]) -> float:
			if not item or item in history:
				return 0
			context_subjects = set(self.current_context(history))
			overlap = set(item.subjects) & context_subjects
			return self.coherence_score(item, overlap)

		def monotonic_bonus(item: Item, monotonic_subjects: list[int]) -> float:
			if not item:
				return 0
			return sum(-1 for s in item.subjects if s in monotonic_subjects)

		# Collect all bonuses
		return [
			freshness_bonus(item, history),
			coherence_bonus(item, history),
			monotonic_bonus(item, monotonic_subjects),
			repetition_bonus(item, history),
			importance_bonus(item, history),
			preference_bonus(item, history),
			pause_bonus(item),
		]

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext, lr=0.1) -> None:
		super().__init__(snapshot, ctx)
		self.lr = lr
		self.baseline = 0.0
		# Load weights if available
		try:
			self.weights = np.load('./players/player_8/learned_weights.npy')
			self.intercept = float(np.load('./players/player_8/learned_intercept.npy'))
		except FileNotFoundError:
			self.weights = np.zeros(7)  # assuming 6 bonus features
			self.intercept = 0.0
		self.last_item = None
		self.last_history = None
		self.last_monotonic = None

	def select_item(self, items, history, monotonic):
		"""Sample item using softmax over weighted bonuses."""

		scores = np.array(
			[
				np.dot(self.weights, self.compute_bonuses(item, history, monotonic))
				for item in items
			],
			dtype=np.float32,
		)

		if scores.shape[0] != len(items):
			raise ValueError(
				f'scores length {scores.shape[0]} does not match items length {len(items)}'
			)

		# softmax safely
		exp_scores = np.exp(scores - np.max(scores))
		probs = exp_scores / np.sum(exp_scores)

		choice_idx = np.random.choice(len(items), p=probs)
		self.last_item = items[choice_idx]
		self.last_history = list(history)  # copy
		self.last_monotonic = list(monotonic)
		return self.last_item

	def update(self, reward):
		"""REINFORCE update using last chosen item."""
		if self.last_item is None:
			return
		grad_log = self.compute_bonuses(
			self.last_item, self.last_history, self.last_monotonic
		)
		self.weights += self.lr * (reward - self.baseline) * np.array(grad_log)
		self.intercept += self.lr * (reward - self.baseline)
		self.baseline = 0.9 * self.baseline + 0.1 * reward

	def propose_item(self, history):
		# Step 1: check if we have a last item to reward
		if len(history) > 0 and history[-1] is not None and history[-1].player_id == self.id:
			reward = self.compute_item_bonus(history[-1], history[:-1], self.monotonic_subjects(history[:-1]))
			self.update(reward)
		
		# Step 2: determine monotonic subjects and candidates
		monotonic_subjects = self.monotonic_subjects(history)
		candidates = list(self.memory_bank) + [None]  # include pause
		# Step 3: propose next item
		if len(history) % 100 == 0:
			np.save("./players/player_8/learned_weights.npy", self.weights)
			np.save("./players/player_8/learned_intercept.npy", self.intercept)
		return self.select_item(candidates, history, monotonic_subjects)
