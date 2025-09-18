from collections import Counter
from models.player import GameContext, Item, Player, PlayerSnapshot
import statistics

class Player8(Player):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:  # noqa: F821
		super().__init__(snapshot, ctx)

	@staticmethod
	def was_last_round_pause(history: list[Item]) -> bool:
		return len(history) >= 1 and history[-1] is None

	@staticmethod
	def get_last_n_subjects(history: list[Item], n: int) -> set[int]:
		return set(
			subject for item in history[-n:] if item is not None for subject in item.subjects
		)

	@staticmethod
	def subjects_from_items(items: list[Item]) -> set[int]:
		return [subject for item in items if item is not None for subject in item.subjects]

	@staticmethod
	def filter_unused(items: list[Item], history: list[Item]) -> list[Item]:
		return [item for item in items if item not in history]

	def get_fresh_items(self, history: list[Item]) -> list[Item]:
		fresh_items = []
		prev_subjects = self.get_last_n_subjects(history, 5)
		for item in self.memory_bank:
			for subject in item.subjects:
				fresh_subject = subject not in prev_subjects
				used = item in history
				if fresh_subject and not used and item not in fresh_items:
					fresh_items.append(item)
		return fresh_items

	def get_most_important_item(self, items: list[Item]) -> Item | None:
		if not items:
			return None
		return max(items, key=lambda item: item.importance)

	"""
		These subjects have already appeared thrice before and are best to avoid, on the current try
	"""
	def monotonic_subjects(self, history: list[Item]) -> list[int]:
		sub1, sub2, sub3 = set(), set(), set()
		monotonic_subjects = []

		if len(history) >= 3:
			counter = Counter(self.subjects_from_items(history[-3:]))
			for x, y in counter.items():
				if y > 2:
					monotonic_subjects.append(x)
		return monotonic_subjects
	

	@staticmethod
	def coherence_score(item: Item, context_subjects: list[Item]):
		coherence_score = 0.0
		for subject in item.subjects:
				item_has_current_subject = subject in context_subjects
				if item_has_current_subject:
					coherence_score += 1
		return coherence_score
	
	def current_context(self, history: list[Item]):
		context_subjects: list[int] = []
		for i in range(-1, -3, -1):
			if len(history) >= -i and history[i]:
				for subject in history[i].subjects:
					context_subjects.append(subject)
			else:
				break

		return context_subjects

	def coherent_items(self, items: list[Item], history: list[Item]) -> list[Item]:
		if not history:
			return items
		context_subjects = self.current_context(history)
		coherent_items = []

		for item in items:
			if self.coherence_score(item, context_subjects) > 0:
				coherent_items.append(item)

		return coherent_items

	def preference_score(self, item: Item):
		return sum(1 - (self.preferences[s] / len(self.preferences)) for s in item.subjects) / len(
						item.subjects
					)

	def get_preferred_item_order(self) -> list[Item]:
		S = len(self.preferences)

		ranked_items = []
		for item in self.memory_bank:

			if not item.subjects:
				continue

			avg_bonus = self.preference_score(item)
			if avg_bonus > 0.5:
				ranked_items.append((avg_bonus, item))

		ranked_items.sort(reverse=True, key=lambda x: x[0])
		return [item for _, item in ranked_items]

	def get_first_unused_item(self, items: list[Item], history: list[Item]) -> Item | None:
		return next(iter(self.filter_unused(items, history)), None)

	@staticmethod
	def filter_monotonic_items(monotonic_subjects: list[int], items: list[Item]) -> list[Item]:
		return [item for item in items if not len(set(monotonic_subjects) & set(item.subjects))]

	def compute_bonuses(
		self,
		item: Item,
		history: list[Item],
		monotonic_subjects: list[int]
	) -> list[float]:

		def repetition_bonus(item: Item, history: list[Item]) -> float:
			return -1 if item and item in history else 0

		def importance_bonus(item: Item, history: list[Item]) -> float:
			return item.importance if item and item not in history else 0

		def preference_bonus(item: Item, history: list[Item]) -> float:
			return self.preference_score(item) if item and item not in history else 0

		def freshness_bonus(item: Item, history: list[Item]) -> float:
			if not item: # pauses contribute to the freshness score
				return 1
			if not self.was_last_round_pause(history):
				return 0
			recent_subjects = self.get_last_n_subjects(history, 6)
			return sum(1 for s in item.subjects if s not in recent_subjects)

		def coherence_bonus(item: Item, history: list[Item]) -> float:
			if not item or item in history:
				return 0
			context_subjects = set(self.current_context(history))
			overlap = set(item.subjects) & context_subjects
			return self.coherence_score(item, overlap)/2

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
		]

	def propose_item(self, history: list[Item]) -> Item | None:
		unused_items = self.filter_unused(self.memory_bank, history)
		preferred_item_order = self.get_preferred_item_order()
		monotonic_subjects = self.monotonic_subjects(history)
		coherent_items = self.coherent_items(unused_items, history)

		candidates = [
			# 1. Most important on-subject
			lambda: self.get_most_important_item(
				self.filter_monotonic_items(monotonic_subjects, coherent_items)
			),
			# 2. Fresh item after pause
			lambda: self.get_most_important_item(self.get_fresh_items(history))
			if self.was_last_round_pause(history)
			else None,
			# 3. Fresh preferred
			lambda: self.get_first_unused_item(
				self.filter_monotonic_items(monotonic_subjects, preferred_item_order),
				history,
			),
			# 4. Most important unused
			lambda: self.get_most_important_item(
				self.filter_monotonic_items(monotonic_subjects, unused_items)
			)
		]

		# Evaluate candidates and compute their bonuses once
		# evaluated_items = [
		# 	(item, self.compute_item_bonus(item, history, monotonic_subjects))
		# 	for candidate in candidates
		# 	if (item := candidate()) is not None
		# ]

		evaluated_items = [
			(item, statistics.mean(self.compute_bonuses(item, history, monotonic_subjects)))
			for candidate in self.memory_bank
			if (item := candidate) is not None
		]

		if not evaluated_items:
			return None

		# Pick the candidate with the highest bonus
		best_item, best_bonus = max(evaluated_items, key=lambda x: x[1])

		return best_item if best_bonus >= 0 else None
