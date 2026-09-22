"""Tests for `src.profiles.attributes`."""

import numpy as np

from src.profiles import attributes


def row(**wins: str) -> np.ndarray:
    """A similarity row where the named prompt of each group scores highest.

    `wins` maps a group name to the label that should win it. An independent
    group takes its own name to say yes, or any other value to say no. Groups
    left out score a tie.
    """
    prompts = attributes.prompts()
    scores = np.full(len(prompts), 0.20)
    for group, label in wins.items():
        if group in attributes.EXCLUSIVE:
            prompt = attributes.EXCLUSIVE[group][label]
        else:
            yes, no = attributes.INDEPENDENT[group]
            prompt = yes if label == group else no
        scores[prompts.index(prompt)] = 0.30
    return scores


class TestPrompts:
    """The prompt list `read` indexes into."""

    def test_covers_every_option_and_every_flag(self):
        """One prompt per exclusive option, two per independent flag."""
        expected = sum(len(o) for o in attributes.EXCLUSIVE.values())
        expected += 2 * len(attributes.INDEPENDENT)
        assert len(attributes.prompts()) == expected

    def test_has_no_duplicates(self):
        """A repeated prompt would make `read` index the wrong group."""
        prompts = attributes.prompts()
        assert len(set(prompts)) == len(prompts)

    def test_says_nothing_about_gender(self):
        """Gender comes from the face model, which is better at it."""
        assert "gender" not in attributes.EXCLUSIVE


class TestRead:
    """Turning one crop's similarities into labelled scores."""

    def test_labels_every_group(self):
        """Each group in the vocabulary gets an entry, labelled or not."""
        labels = attributes.read(row())
        assert set(labels) == {*attributes.EXCLUSIVE, *attributes.INDEPENDENT}

    def test_picks_the_best_scoring_option_in_a_group(self):
        """The winning prompt names the group."""
        assert attributes.read(row(headwear="cap"))["headwear"][0] == "cap"

    def test_a_winning_positive_names_the_flag(self):
        """An independent group is labelled with its own name."""
        assert attributes.read(row(scarf="scarf"))["scarf"][0] == "scarf"

    def test_a_winning_negative_leaves_the_flag_unlabelled(self):
        """The negative prompt means the flag is absent, not unknown."""
        assert attributes.read(row(scarf="none"))["scarf"][0] is None

    def test_scores_are_probabilities(self):
        """Softmax within a group keeps every score in 0..1."""
        for _, score in attributes.read(row(headwear="cap")).values():
            assert 0.0 <= score <= 1.0

    def test_a_tie_splits_a_flag_evenly(self):
        """Equal prompts give no reason to prefer either side."""
        assert attributes.read(row())["glasses"][1] == 0.5

    def test_a_clear_win_scores_above_a_tie(self):
        """Confidence is what `--min-confidence` filters on, so it must move."""
        decided = attributes.read(row(headwear="cap"))["headwear"][1]
        assert decided > attributes.read(row())["headwear"][1]

    def test_groups_are_decided_independently(self):
        """Winning one group does not disturb the next."""
        labels = attributes.read(row(headwear="beanie", hair="bald"))
        assert labels["headwear"][0] == "beanie"
        assert labels["hair"][0] == "bald"
