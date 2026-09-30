"""UI-independent Target definition, realization, and edit-session boundary."""

from .definition import (
    canonical_target_definition,
    target_definition_frd_bytes,
    target_definition_from_asset,
)
from .presets import (
    TARGET_PRESET_SCHEMA_VERSION,
    TARGET_PRESET_SCOPES,
    TARGET_PRESET_CATEGORIES,
    normalize_target_preset,
    shared_target_preset_catalog,
    materialize_target_preset,
    get_shared_target_preset,
    shared_target_preset_paths,
)
from .exchange import (
    TargetEditSession,
    create_target_edit_session,
    read_target_edit_session,
    save_target_edit_result,
)
from .generator import (
    TargetProcessingResult,
    TargetProcessingSettings,
    flat_target_response,
    process_target,
)
from .response import (
    apply_target_edit_payload,
    apply_target_edit_to_response,
    normalize_target_edit_payload,
)

__all__ = [
    "TargetEditSession",
    "TargetProcessingResult",
    "TargetProcessingSettings",
    "TARGET_PRESET_SCHEMA_VERSION",
    "TARGET_PRESET_SCOPES",
    "TARGET_PRESET_CATEGORIES",
    "apply_target_edit_payload",
    "apply_target_edit_to_response",
    "canonical_target_definition",
    "create_target_edit_session",
    "flat_target_response",
    "normalize_target_preset",
    "normalize_target_edit_payload",
    "read_target_edit_session",
    "process_target",
    "save_target_edit_result",
    "shared_target_preset_catalog",
    "materialize_target_preset",
    "get_shared_target_preset",
    "shared_target_preset_paths",
    "target_definition_frd_bytes",
    "target_definition_from_asset",
]
from .repository import (
    archive_target,
    ResolvedTargetBinding,
    TargetBinding,
    TargetRevisionRecord,
    TargetRevisionRef,
    TargetSummary,
    create_target,
    effective_target_binding,
    ensure_target_repository,
    find_target_revision_by_hash,
    get_target_revision,
    list_targets,
    list_target_revisions,
    list_target_revision_summaries,
    read_changes,
    resolve_target_binding,
    restore_target_as_new_revision,
    save_target_binding,
    save_target_revision,
)

__all__ += [
    "archive_target",
    "ResolvedTargetBinding",
    "TargetBinding",
    "TargetRevisionRecord",
    "TargetRevisionRef",
    "TargetSummary",
    "create_target",
    "effective_target_binding",
    "ensure_target_repository",
    "find_target_revision_by_hash",
    "get_target_revision",
    "list_targets",
    "list_target_revisions",
    "list_target_revision_summaries",
    "read_changes",
    "resolve_target_binding",
    "restore_target_as_new_revision",
    "save_target_binding",
    "save_target_revision",
]
