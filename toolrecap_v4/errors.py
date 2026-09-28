"""Error hierarchy for toolrecap_v4."""

from __future__ import annotations

from typing import Optional


class ToolRecapError(Exception):
    """Base exception for all toolrecap_v4 errors."""


class ValidationError(ToolRecapError):
    """Base exception for validation failures."""


class SchemaValidationError(ValidationError):
    """Raised when JSON schema validation fails."""


JSON_SOURCE_NOT_FOUND = "JSON_SOURCE_NOT_FOUND"


class JsonSourceNotFoundError(SchemaValidationError):
    """Raised when a source file in JSON is not found in actual source mapping or exact basename mismatch."""

    code: str = JSON_SOURCE_NOT_FOUND
    category: str = JSON_SOURCE_NOT_FOUND

    def __init__(
        self,
        message: str,
        category: str = JSON_SOURCE_NOT_FOUND,
        code: str = JSON_SOURCE_NOT_FOUND,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.code = code


SourceNotFoundError = JsonSourceNotFoundError


class SourceChangedError(ValidationError):
    """Raised when source video file has been modified since project creation."""


class WindowsNameError(ValidationError):
    """Base exception for Windows naming and filesystem constraints."""


class WindowsReservedNameError(WindowsNameError):
    """Raised when a name or title matches Windows reserved device names."""


class WindowsCollisionError(WindowsNameError):
    """Raised when names or identifiers collide under case-folding."""


class InvalidCharacterError(WindowsNameError):
    """Raised when invalid Windows filename characters are detected."""


class DuplicateIdError(ValidationError):
    """Raised when duplicate IDs are found within the same scope."""


class TimestampBoundaryError(ValidationError):
    """Raised when timestamps are invalid, non-finite, out of order, or exceed duration."""


class NarrationFitError(ValidationError):
    """Raised when narration duration exceeds visual segment duration.
    
    Fit policy is defined as a technical error; trimming or altering
    editorial clips is strictly prohibited.
    """


class DiscoveryError(ToolRecapError):
    """Raised when source discovery fails."""


class AnalysisPipelineUnavailableError(ToolRecapError):
    """Raised when analysis is required but the analysis pipeline is unavailable in Phase 2."""


class ScannerError(ToolRecapError):
    """Base exception for factual transcript Scanner failures."""

    def __init__(
        self,
        message: str,
        *,
        episode_id: str | None = None,
        chunk_id: str | None = None,
        request_phase: str | None = None,
        retry_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.episode_id = episode_id
        self.chunk_id = chunk_id
        self.request_phase = request_phase
        self.retry_count = retry_count


class ScannerChunkPlanningError(ScannerError):
    """Raised when a transcript cannot be converted into valid Scanner chunks."""


class ScannerCapacityError(ScannerChunkPlanningError):
    """Raised when even a losslessly split transcript part cannot fit the request envelope."""


class ScannerResponseError(ScannerError):
    """Raised when a Scanner response cannot be parsed as structured JSON."""


class ScannerValidationError(ScannerResponseError):
    """Raised when structured Scanner output violates the factual contract."""

    def __init__(self, message: str, *, issue_codes: tuple[str, ...] = (), **context: object) -> None:
        super().__init__(message, **context)
        self.issue_codes = issue_codes


class ScannerRepairExhaustedError(ScannerError):
    """Raised after bounded technical repair attempts fail for one Scanner chunk."""

    def __init__(self, message: str, *, validation_errors: tuple[str, ...] = (), **context: object) -> None:
        super().__init__(message, **context)
        self.validation_errors = validation_errors


class EvidenceStoreError(ToolRecapError):
    """Raised when evidence persistence or integrity verification fails."""


class EvidenceRevisionError(EvidenceStoreError):
    """Raised for an unknown, incomplete, or dependency-mismatched evidence revision."""


class CatalogError(ToolRecapError):
    """Base exception for deterministic Season Evidence Catalog failures."""


class CatalogValidationError(CatalogError):
    """Raised when a Catalog violates identity, coverage, ordering, or hash rules."""

    def __init__(self, message: str, *, issue_codes: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.issue_codes = issue_codes


class CatalogPackingError(CatalogError):
    """Raised when packed Catalog data is malformed or not losslessly reversible."""


class CatalogStoreError(CatalogError):
    """Raised for Catalog checkpoint persistence or integrity failures."""


class CatalogCapacityError(CatalogError):
    """Raised for invalid explicit Catalog capacity configuration."""


class PlannerError(ToolRecapError):
    """Base exception for Season Planner failures with round context."""

    def __init__(self, message: str, *, project_id: str | None = None, round_id: str | None = None) -> None:
        super().__init__(message)
        self.project_id = project_id
        self.round_id = round_id


class PlannerCapacityError(PlannerError):
    """Raised when a complete Planner request exceeds explicit or upstream capacity."""


class PlannerProtocolError(PlannerError):
    """Raised for an unsupported Planner protocol or action."""


class PlannerResponseError(PlannerError):
    """Raised when Planner output cannot be parsed as a structured response."""


class PlannerValidationError(PlannerProtocolError):
    """Raised when structured Planner output violates the technical contract."""

    def __init__(self, message: str, *, issue_codes: tuple[str, ...] = (), **context: object) -> None:
        super().__init__(message, **context)
        self.issue_codes = issue_codes


class PlannerRepairExhaustedError(PlannerError):
    """Raised when bounded technical repair cannot validate a Planner round."""


class PlannerRoundLimitError(PlannerError):
    """Raised when Planner asks for more evidence after the final allowed round."""


class PlannerEvidenceRequestError(PlannerError):
    """Base exception for invalid exact Evidence requests."""


class PlannerEvidenceNotFoundError(PlannerEvidenceRequestError):
    """Raised for an unknown or malformed requested Evidence ID."""


class PlannerEvidenceRevisionError(PlannerEvidenceRequestError):
    """Raised when a request targets a stale Evidence revision."""


class PlannerEvidenceIntegrityError(PlannerEvidenceRequestError):
    """Raised when Catalog detail identity disagrees with Full Evidence."""


class PlannerSessionError(PlannerError):
    """Raised for Planner checkpoint persistence or resume failures."""


class VisualEvidenceError(ToolRecapError):
    """Base exception for selective visual evidence failures."""


class VisualRequestError(VisualEvidenceError): pass
class FrameExtractionError(VisualEvidenceError): pass
class FrameBudgetError(VisualEvidenceError): pass
class VisionResponseError(VisualEvidenceError): pass
class VisionValidationError(VisionResponseError): pass
class VisionRepairExhaustedError(VisualEvidenceError): pass
class VisualStoreError(VisualEvidenceError): pass


class SeasonPlanError(ToolRecapError): pass
class FinalPlannerValidationError(SeasonPlanError):
    def __init__(self, message: str, *, issue_codes: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.issue_codes = issue_codes
class SeasonPlanLockError(SeasonPlanError): pass


class WriterError(ToolRecapError): pass
class WriterContextError(WriterError): pass
class WriterEvidenceError(WriterContextError): pass
class WriterVisualEvidenceError(WriterContextError): pass
class WriterCapacityError(WriterError): pass
class WriterTransportError(WriterError): pass
class WriterResponseError(WriterError): pass
class WriterPersistenceError(WriterError): pass
class WriterRevisionError(WriterError): pass


class WriterValidationError(ToolRecapError):
    def __init__(self,message:str,*,output_id:str|None=None,issue_codes:tuple[str,...]=()):super().__init__(message);self.output_id=output_id;self.issue_codes=issue_codes
class WriterReferenceError(WriterValidationError): pass
class WriterRepairError(ToolRecapError): pass
class WriterRepairExhaustedError(WriterRepairError): pass
class FinalJsonMappingError(ToolRecapError): pass
class FinalJsonValidationError(ToolRecapError): pass
class FinalJsonPersistenceError(ToolRecapError): pass
class FinalJsonRevisionError(ToolRecapError): pass
class ZeroOutputError(ToolRecapError): pass
class OneShotRoutingError(ToolRecapError): pass


class CancelledError(ToolRecapError):
    """Raised when an operation is cancelled."""


class PersistenceError(ToolRecapError):
    """Base exception for storage and persistence failures."""


class SecretExposureError(PersistenceError):
    """Raised when an attempt is made to store secrets in project state."""


class DPAPIError(PersistenceError):
    """Raised when Windows DPAPI operation fails or DPAPI is unavailable."""


class SecretStorageError(ToolRecapError):
    """Raised when secret storage operation fails."""


class GatewayError(ToolRecapError):
    """Base exception for AI Gateway client failures."""


class GatewayConnectionError(GatewayError):
    """Raised when network connection to AI Gateway fails."""


class GatewayTimeoutError(GatewayError):
    """Raised when request to AI Gateway times out."""


class GatewayRequestTimeoutError(GatewayTimeoutError):
    """Raised when Gateway returns HTTP 408 Request Timeout."""

    def __init__(self, message: str, status_code: int = 408) -> None:
        super().__init__(message)
        self.status_code = status_code


class GatewayAuthenticationError(GatewayError):
    """Raised when Gateway returns HTTP 401 Unauthorized."""

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


class GatewayPermissionError(GatewayError):
    """Raised when Gateway returns HTTP 403 Forbidden."""

    def __init__(self, message: str, status_code: int = 403) -> None:
        super().__init__(message)
        self.status_code = status_code


class GatewayNotFoundError(GatewayError):
    """Raised when Gateway returns HTTP 404 Not Found."""

    def __init__(self, message: str, status_code: int = 404) -> None:
        super().__init__(message)
        self.status_code = status_code


class GatewayRateLimitError(GatewayError):
    """Raised when Gateway returns HTTP 429 Rate Limit Exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        status_code: int = 429,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code


class GatewayServerError(GatewayError):
    """Raised when Gateway returns HTTP 5xx Server Error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class PayloadContextError(GatewayError):
    """Raised when request payload exceeds context window or prompt size limits."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnsupportedMediaError(GatewayError):
    """Raised when media format, container, dimensions, or size fails validation."""


class GatewayResponseError(GatewayError):
    """Raised when Gateway returns an error or malformed response."""

    def __init__(
        self,
        message: str,
        raw_response: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.status_code = status_code


class MalformedApiResponseError(GatewayResponseError):
    """Raised when Gateway API HTTP response envelope is malformed or invalid JSON/SSE."""


class EmptyApiResponseError(GatewayResponseError):
    """Raised when Gateway returns an empty response body or empty choices."""


class InvalidGatewayResponseError(GatewayResponseError):
    """Raised when Gateway response cannot be parsed as JSON (compatibility base)."""


class MalformedModelJsonError(InvalidGatewayResponseError):
    """Raised when Gateway model output text cannot be parsed as JSON."""


class ModelCapabilityError(GatewayError):
    """Raised when model lacks requested capability."""


class VoiceStudioError(ToolRecapError):
    """Base exception for VoiceStudio adapter failures."""


class VoiceStudioUnavailableError(VoiceStudioError):
    """Raised when VoiceStudio (local and/or remote) is unavailable."""


class SourceDialogueNarrationError(VoiceStudioError):
    """Raised before TTS when source character dialogue is routed as narration."""


class InvalidAudioError(VoiceStudioError):
    """Raised when synthesized audio fails WAV verification or duration check."""


class UpdaterError(ToolRecapError):
    """Base exception for updater failures."""


class UpdateNotConfiguredError(UpdaterError):
    """Raised when updater repository is not configured."""


class UpdateCheckError(UpdaterError):
    """Raised when checking for updates fails."""


class UpdateVerificationError(UpdaterError):
    """Raised when update verification (checksum) fails."""


class ChecksumMismatchError(UpdateVerificationError):
    """Raised when update checksum does not match sha256 file."""


class MaliciousArchiveError(UpdaterError):
    """Raised when update archive contains traversal, symlink, or zip bomb."""


class PackageValidationError(UpdaterError):
    """Raised when extracted update package is missing required files or invalid marker."""


class UpdateInProgressError(UpdaterError):
    """Raised when an update operation is already active."""


class UpdateApplyError(UpdaterError):
    """Raised when applying update fails."""
