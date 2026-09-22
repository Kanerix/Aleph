"""Turn each profile and its closest reference into a manifest entry."""

from .matching import best


def _identity(profile, vector, names, references, threshold: float) -> dict:
    ranked = best(vector, references)
    row, score = ranked[0]
    entry = {
        "id": profile["id"],
        "appearances": profile["appearances"],
        "photos": profile["photos"],
        "match": names[row] if score >= threshold else None,
        "score": round(score, 3),
        "runner_up": None,
    }
    for row, score in ranked[1:]:
        entry["runner_up"] = {"name": names[row], "score": round(score, 3)}
    return entry


def identify(profiles, vectors, names, references, threshold: float) -> list[dict]:
    """Name each profile after its closest reference, if it is close enough.

    Every profile gets an entry, matched or not, because the manifest is what
    joins this stage to whatever reads it. An unmatched profile still records
    the score it reached, so a threshold that is slightly too high is visible
    rather than silent.
    """
    return [
        _identity(profile, vector, names, references, threshold)
        for profile, vector in zip(profiles, vectors, strict=True)
    ]
