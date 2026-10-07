from __future__ import annotations

import json
from typing import Any
import sys
from pathlib import Path

import ollama
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.agents"
from ..prompt.router_agent_prompt import ROUTER_PROMPT
from .no_tools_calling_agent import SimpleAgent


class RouterAgent(SimpleAgent):
    """Classify research queries into validated research strategies.

    The router does not perform research itself. It sends the user's query to
    an Ollama model, requests a JSON classification, validates the returned
    object, and provides the resulting strategy to the research orchestrator.

    The returned configuration contains exactly six fields:

    ``research_depth``
        Required research depth: ``shallow``, ``medium``, or ``deep``.

    ``freshness``
        Required information freshness: ``low``, ``medium``, or ``high``.

    ``comparison``
        Whether the query explicitly compares multiple entities or options.

    ``technical``
        Whether the query is primarily technical.

    ``academic``
        Whether academic or scientific sources are central to the request.

    ``verification``
        Whether strong verification or multiple independent sources are
        required.

    Args:
        model: Ollama model name used for query classification.
        temperature: Sampling temperature. A value close to zero is recommended
            for deterministic routing.

    Raises:
        TypeError: If the query is not a string.
        ValueError: If the query is empty or the model returns an invalid
            configuration.
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
        """Classify a research query.

        This method uses a non-streaming Ollama request because the complete
        JSON object must be validated before it is returned.

        Args:
            query: Original research query.

        Returns:
            A validated dictionary containing exactly the router fields.

        Raises:
            TypeError: If ``query`` is not a string.
            ValueError: If the query is empty, the response is invalid, or the
                returned configuration does not match the schema.
        """
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

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

        if not isinstance(config, dict):
            raise ValueError(
                "RouterAgent response must be a JSON object."
            )

        return self._validate(
            config
        )

    @staticmethod
    def _extract_content(
        response: Any,
    ) -> str:
        """Extract non-empty assistant content from an Ollama response.

        Args:
            response: Raw response returned by ``ollama.chat``.

        Returns:
            Non-empty response content.

        Raises:
            ValueError: If the response does not contain valid message
                content.
        """
        if isinstance(response, dict):
            message = response.get(
                "message",
                {},
            )

            if isinstance(message, dict):
                content = message.get(
                    "content",
                    "",
                )

                if isinstance(
                    content,
                    str,
                ) and content.strip():
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

        if isinstance(
            content,
            str,
        ) and content.strip():
            return content.strip()

        raise ValueError(
            "RouterAgent received an empty or invalid model response."
        )

    def _validate(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate a decoded router configuration.

        Args:
            config: JSON object returned by the model.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If fields are missing, unexpected, invalid, or have
                incorrect types.
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

        return config
    
if __name__ == '__main__':
    agent = RouterAgent()
    res = agent.route("Qui est ninho")
    print(res)