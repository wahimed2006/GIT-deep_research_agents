from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama


if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        str(
            Path(__file__).resolve().parents[2]
        ),
    )
    __package__ = "src.agents"


from ..prompt.router_agent_prompt import ROUTER_PROMPT
from .no_tools_calling_agent import SimpleAgent


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Configuration controlling the research pipeline.

    Attributes:
        research_depth: Required research depth.
        freshness: Required information freshness.
        comparison: Whether the query compares multiple entities.
        technical: Whether the query is primarily technical.
        academic: Whether academic sources are central.
        verification: Whether strong source verification is required.
    """

    research_depth: str
    freshness: str
    comparison: bool
    technical: bool
    academic: bool
    verification: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-compatible dictionary."""
        return {
            "research_depth": self.research_depth,
            "freshness": self.freshness,
            "comparison": self.comparison,
            "technical": self.technical,
            "academic": self.academic,
            "verification": self.verification,
        }


class RouterAgent(SimpleAgent):
    """Classify research queries into validated research strategies.

    The router does not perform research itself. It sends the user's query to
    an Ollama model, requests a JSON classification, validates the returned
    object, and provides the resulting strategy to the research orchestrator.

    The model must return exactly six fields:

    ``research_depth``
        ``shallow``, ``medium``, or ``deep``.

    ``freshness``
        ``low``, ``medium``, or ``high``.

    ``comparison``
        Whether the query explicitly compares multiple entities or options.

    ``technical``
        Whether the query is primarily technical.

    ``academic``
        Whether academic or scientific sources are central.

    ``verification``
        Whether strong verification or multiple independent sources are
        required.

    Args:
        model: Ollama model name used for classification.
        temperature: Sampling temperature. A value close to zero is recommended
            for deterministic routing.
    """

    VALID_DEPTHS = frozenset(
        {
            "shallow",
            "medium",
            "deep",
        }
    )

    VALID_FRESHNESS = frozenset(
        {
            "low",
            "medium",
            "high",
        }
    )

    REQUIRED_FIELDS = frozenset(
        {
            "research_depth",
            "freshness",
            "comparison",
            "technical",
            "academic",
            "verification",
        }
    )

    BOOLEAN_FIELDS = (
        "comparison",
        "technical",
        "academic",
        "verification",
    )

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
    ) -> None:
        """Initialize the router agent.

        Args:
            model: Ollama model name.
            temperature: Sampling temperature used by Ollama.
        """
        super().__init__(
            model_name=model,
            system_prompt=ROUTER_PROMPT,
        )

        self.temperature = temperature

    def route(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Classify a research query and return a validated dictionary.

        Args:
            query: Original research query.

        Returns:
            A dictionary containing exactly the six router fields.

        Raises:
            TypeError: If ``query`` is not a string.
            ValueError: If the query is empty, the model response is invalid,
                or the configuration violates the schema.
            RuntimeError: If the Ollama request fails.
        """
        config = self.route_config(
            query
        )

        return config.to_dict()

    def route_config(
        self,
        query: str,
    ) -> ResearchConfig:
        """Classify a query and return a validated ``ResearchConfig``.

        Args:
            query: Original research query.

        Returns:
            A validated research configuration.
        """
        normalized_query = self._validate_query(
            query
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": ROUTER_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": normalized_query,
                    },
                ],
                stream=False,
                options={
                    "temperature": self.temperature,
                },
                format="json",
            )
        except Exception as error:
            raise RuntimeError(
                "RouterAgent could not call Ollama."
            ) from error

        content = self._extract_content(
            response
        )

        try:
            config = json.loads(
                content
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "RouterAgent returned invalid JSON: "
                f"{content}"
            ) from error

        if not isinstance(
            config,
            dict,
        ):
            raise ValueError(
                "RouterAgent response must be a JSON object."
            )

        return self._validate_config(
            config
        )

    @staticmethod
    def _validate_query(
        query: str,
    ) -> str:
        """Validate and normalize a user query.

        Args:
            query: Query to validate.

        Returns:
            A stripped query.

        Raises:
            TypeError: If the query is not a string.
            ValueError: If the query is empty.
        """
        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

        return normalized_query

    @staticmethod
    def _extract_content(
        response: Any,
    ) -> str:
        """Extract non-empty assistant content from an Ollama response.

        Args:
            response: Raw response returned by Ollama.

        Returns:
            Non-empty response content.

        Raises:
            ValueError: If no valid message content is available.
        """
        if isinstance(
            response,
            dict,
        ):
            message = response.get(
                "message",
                {},
            )

            if isinstance(
                message,
                dict,
            ):
                content = message.get(
                    "content",
                    "",
                )

                if (
                    isinstance(
                        content,
                        str,
                    )
                    and content.strip()
                ):
                    return content.strip()

        message = getattr(
            response,
            "message",
            None,
        )

        content = getattr(
            message,
            "content",
            "",
        )

        if (
            isinstance(
                content,
                str,
            )
            and content.strip()
        ):
            return content.strip()

        raise ValueError(
            "RouterAgent received an empty or invalid model response."
        )

    def _validate_config(
        self,
        config: dict[str, Any],
    ) -> ResearchConfig:
        """Validate a decoded JSON configuration.

        Args:
            config: JSON object returned by the model.

        Returns:
            A validated ``ResearchConfig``.

        Raises:
            ValueError: If fields or values are invalid.
        """
        received_fields = frozenset(
            config.keys()
        )

        if received_fields != self.REQUIRED_FIELDS:
            raise ValueError(
                "Invalid router configuration. "
                f"Expected fields: "
                f"{sorted(self.REQUIRED_FIELDS)}. "
                f"Received fields: "
                f"{sorted(received_fields)}"
            )

        research_depth = config[
            "research_depth"
        ]

        if not isinstance(
            research_depth,
            str,
        ):
            raise ValueError(
                "research_depth must be a string."
            )

        if research_depth not in self.VALID_DEPTHS:
            raise ValueError(
                "Invalid research_depth: "
                f"{research_depth}"
            )

        freshness = config[
            "freshness"
        ]

        if not isinstance(
            freshness,
            str,
        ):
            raise ValueError(
                "freshness must be a string."
            )

        if freshness not in self.VALID_FRESHNESS:
            raise ValueError(
                "Invalid freshness: "
                f"{freshness}"
            )

        for field in self.BOOLEAN_FIELDS:
            if type(config[field]) is not bool:
                raise ValueError(
                    f"{field} must be a JSON boolean."
                )

        return ResearchConfig(
            research_depth=research_depth,
            freshness=freshness,
            comparison=config[
                "comparison"
            ],
            technical=config[
                "technical"
            ],
            academic=config[
                "academic"
            ],
            verification=config[
                "verification"
            ],
        )


if __name__ == "__main__":
    agent = RouterAgent()

    try:
        result = agent.route_config(
            "RTX 5090 vs RTX 4090"
        )
    except (RuntimeError, ValueError, TypeError) as error:
        print(
            f"Router error: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )