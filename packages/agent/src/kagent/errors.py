"""Public Agent core errors."""


class AgentError(Exception):
    """Base class for fatal harness errors."""


class AgentBusyError(AgentError):
    """Raised when a second run is started on a busy agent."""


class MaxTurnsExceeded(AgentError):
    """Raised when a run consumes its configured model-turn budget."""
