from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


class ToolOutput(ABC):
    @abstractmethod
    def get_response(self) -> str:
        ...

    @abstractmethod
    def get_error(self) -> Optional[str]:
        ...


class Tool(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def execute(self, input) -> ToolOutput:
        ...

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def system_prompt_fragment(self) -> str:
        ...


@dataclass
class StringToolOutput(ToolOutput):
    response: str

    def get_response(self) -> str:
        return self.response

    def get_error(self) -> Optional[str]:
        return None
