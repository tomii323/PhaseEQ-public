"""UI/DB/DSP-independent contracts for list menus."""
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any


@dataclass(frozen=True)
class ListPolicy:
    edit_limit: int = 128
    candidate_limit: int = 8192

    def __post_init__(self):
        if self.edit_limit < 1 or self.candidate_limit < 1:
            raise ValueError('List limits must be positive')


# Overrides are keyed by stable list identity, not translated labels.
LIST_POLICIES: dict[str, ListPolicy] = {}
DEFAULT_POLICY = ListPolicy()


def list_policy(list_id: str) -> ListPolicy:
    return LIST_POLICIES.get(list_id, DEFAULT_POLICY)


def require_capacity(list_id: str, count: int, additional: int = 1) -> None:
    limit = list_policy(list_id).edit_limit
    if additional < 0:
        raise ValueError('additional must be nonnegative')
    if additional and count + additional > limit:
        raise ValueError(f'{list_id}: 上限{limit}件のため追加できません（現在{count}件）。')


def resolve_choice(options: Sequence[Any], current: Any, default: Any = None, *, required=True):
    if not options:
        return None
    if current in options:
        return current
    if not required:
        return None
    return default if default in options else options[0]


def candidate_slice(records: Sequence[Any], list_id: str, page: int = 0, *, page_size: int | None = None):
    """Bound one candidate batch without deleting or truncating its source."""
    limit = list_policy(list_id).candidate_limit
    if page_size is not None:
        limit = min(limit, max(1, int(page_size)))
    pages = max(1, (len(records) + limit - 1) // limit)
    page = min(max(int(page), 0), pages - 1)
    return records[page * limit:(page + 1) * limit], page, pages
