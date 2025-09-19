from collections import Counter
from uuid import UUID
from models.player import GameContext, Item, Player, PlayerSnapshot


class Player8Base(Player):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:  # noqa: F821
		snapshot.item_in_memory_bank
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

	@staticmethod
	def play_probability(item: Item, history: list[Item], number_of_players: int, player_id: UUID):
		if not item:
			return 0
		if not history:
			return 1 / number_of_players

		items_contributed = Player8Base.compute_player_counts(history)
		if not items_contributed:
			return 1 / number_of_players

		_, min_contri_score = min(items_contributed.items(), key=lambda x: x[1])
		min_contri_players = [
			player_id
			for player_id, contrib in items_contributed.items()
			if contrib == min_contri_score
		]
		res = 1001
		if player_id in min_contri_players:
			res = 1 / len(min_contri_players)
		if history[-1] and player_id == history[-1].player_id:
			res = max(0.5, res)

		if res == 1001:
			res = 1 / (number_of_players)
		return res

	@staticmethod
	def compute_bonuses(
		item: Item, history: list[Item], monotonic_subjects: list[int], preferences: list[int]
	) -> list[float]:

		if not item:  # pause shortcut
			return [1, 0, 0, 0, 0, 0]

		def repetition_bonus():
			return -2 if item in history else 0

		def preference_bonus():
			return Player8Base.preference_score(preferences, item)

		def importance_bonus():
			return item.importance

		def freshness_bonus():
			if not Player8Base.was_last_round_pause(history):
				return 0
			prev_subjects = Player8Base.get_last_n_subjects(history, 6)
			for s in item.subjects:
				if s not in prev_subjects:
					return 1
			return 0

		def coherence_bonus():
			context_subjects = set(Player8Base.current_context(history))
			return Player8Base.coherence_score(item, context_subjects)

		def monotonic_bonus():
			if not item:
				return 0
			return sum(-1 for s in item.subjects if s in monotonic_subjects)

		return [
			freshness_bonus() * 0.8,
			coherence_bonus() * 1.2,
			monotonic_bonus() * 5,
			repetition_bonus() * 5,
			importance_bonus() * 0.5,
			preference_bonus() * 0.5,
		]

	@staticmethod
	def compute_player_counts(history: list[Item]) -> dict[UUID, int]:
		counts = {}
		for item in history:
			if item is None:
				continue
			pid = item.player_id
			counts[pid] = counts.get(pid, 0) + 1
		return counts

	@staticmethod
	def get_most_important_item(items: list[Item]) -> Item | None:
		if not items:
			return None
		return max(items, key=lambda item: item.importance)
	

	@staticmethod
	def monotonic_subjects(history: list[Item]) -> list[int]:
		monotonic_subjects = []

		if len(history) >= 3:
			counter = Counter(Player8Base.subjects_from_items(history[-3:]))
			for x, y in counter.items():
				if y > 2:
					monotonic_subjects.append(x)
		return monotonic_subjects

	@staticmethod
	def coherence_score(item: Item, context_subjects: list[Item]):
		coherence_score = -1
		for subject in item.subjects:
			item_has_current_subject = subject in context_subjects
			if item_has_current_subject:
				coherence_score += 1
		return coherence_score

	@staticmethod
	def current_context(history: list[Item]):
		context_subjects: list[int] = []
		for i in range(-1, -3, -1):
			if len(history) >= -i and history[i]:
				for subject in history[i].subjects:
					context_subjects.append(subject)
			else:
				break

		return context_subjects

	@staticmethod
	def preference_score(preferences: list[int], item: Item):
		return sum(1 - (preferences[s] / len(preferences)) for s in item.subjects) / len(
			item.subjects
		)

	@staticmethod
	def get_first_unused_item( items: list[Item], history: list[Item]) -> Item | None:
		return next(iter(Player8Base.filter_unused(items, history)), None)

	@staticmethod
	def filter_monotonic_items(monotonic_subjects: list[int], items: list[Item]) -> list[Item]:
		return [item for item in items if not len(set(monotonic_subjects) & set(item.subjects))]

	def propose_item(self, history: list[Item]) -> Item | None:

		if None not in self.memory_bank:
			self.memory_bank.append(None)

		monotonic_subjects = self.monotonic_subjects(history)

		evaluated_items = [
			(item, sum(self.compute_bonuses(item, history, monotonic_subjects, self.preferences)))
			for candidate in self.memory_bank
			if (item := candidate) is not None
		]

		best_item, best_bonus = max(evaluated_items, key=lambda x: x[1])

		return best_item if best_bonus >= 0.0 else None
