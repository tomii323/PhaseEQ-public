"""Persisted DSP destination selection, independent of Streamlit and design state."""
from dataclasses import dataclass

from .adapters.registry import available_adapters
from .cdsp_dirac import PROFILE_ID


@dataclass(frozen=True)
class DSPDestination:
    destination_id: str
    display_name: str
    adapter: object
    profiles: tuple


def available_destinations():
    result = []
    for adapter in available_adapters():
        profiles = adapter.profiles()
        standard = tuple(profile for profile in profiles if profile.profile_id != PROFILE_ID)
        if standard:
            result.append(DSPDestination(adapter.adapter_id, adapter.display_name, adapter, standard))
        for profile in profiles:
            if profile.profile_id == PROFILE_ID:
                result.append(DSPDestination(profile.profile_id, profile.display_name, adapter, (profile,)))
    return tuple(result)


def normalize_destination(raw):
    raw = raw if isinstance(raw, dict) else {}
    destinations = available_destinations()
    ids = {item.destination_id for item in destinations}
    selected = raw.get('destination_id')
    if not isinstance(selected, str) or selected not in ids:
        selected = destinations[0].destination_id
    saved = raw.get('profile_ids', {})
    saved = saved if isinstance(saved, dict) else {}
    profiles = {}
    for destination in destinations:
        valid = {profile.profile_id for profile in destination.profiles}
        value = saved.get(destination.destination_id)
        profiles[destination.destination_id] = value if isinstance(value, str) and value in valid else destination.profiles[0].profile_id
    return {'destination_id': selected, 'profile_ids': profiles}


def selected_destination(raw):
    normalized = normalize_destination(raw)
    destination = next(item for item in available_destinations()
                       if item.destination_id == normalized['destination_id'])
    profile = next(item for item in destination.profiles
                   if item.profile_id == normalized['profile_ids'][destination.destination_id])
    return destination, profile
