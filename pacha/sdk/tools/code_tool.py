import abc
from dataclasses import dataclass, field
from typing import Optional
from pacha.data_engine.catalog import Catalog, render_catalog
from pacha.data_engine.data_engine import DataEngine
from pacha.query_planner import QueryPlanner, QueryPlanningInput, UserTurn
from pacha.query_planner.data_context import SqlStatement
from pacha.query_planner.python_executor import PythonExecutor, PythonExecutorHooks
from pacha.sdk.tools.sql_tool import SYSTEM_PROMPT_FRAGMENT_TEMPLATE
from pacha.sdk.tools.tool import Tool, ToolOutput

CODE_ARGUMENT_NAME = "python_code"

TOOL_DESCRIPTION = """
This tool can be used to write Python scripts to retrieve data from the user's database.

The way to retrieve data is to use the `executor.get_from_database` Python function, which takes as input
a SQL query and returns a list of rows, with each row represented as a dictionary of column names to column values.
Always account for the possibilty of rows not meeting your filters in your python code or nullable columns returning None.
The way to output anything from the python script is to use the `executor.output` Python function which takes a string input.
"""

CODE_ARGUMENT_DESCRIPTION = """
The python code to execute. This must be Python code and not direct SQL. Examples:

```
data = executor.get_from_database("SELECT title FROM library.articles WHERE id = 5")
if len(data) == 0:
  executor.output('not found')
else:
  executor.output(f'{data[0]["title"]}')
```

```
data = executor.get_from_database("SELECT MIN(date) AS min_date FROM library.articles WHERE EXTRACT(YEAR FROM date) = 2023")
min_date = data[0][min_date]
executor.output(f'{min_date'})
```
"""


@dataclass
class PythonToolOutput(ToolOutput):
    output: str
    error: Optional[str]
    sql_statements: list[SqlStatement]

    def get_response(self) -> str:
        response = self.output
        if self.error is not None:
            response += self.error
        return response
    
    def get_error(self) -> Optional[str]:
        return self.error


@dataclass
class PachaPythonTool(Tool):
    data_engine: DataEngine
    hooks: PythonExecutorHooks = field(default_factory=PythonExecutorHooks)
    catalog: Catalog = field(init=False)

    def __post_init__(self):
        self.catalog = self.data_engine.get_catalog()

    def execute(self, input) -> PythonToolOutput:
        input_code = input[CODE_ARGUMENT_NAME]
        executor = PythonExecutor(
            data_engine=self.data_engine, hooks=self.hooks)
        executor.exec_code(input_code)
        return PythonToolOutput(output=executor.output_text, error=executor.error, sql_statements=executor.sql_statements)

    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                CODE_ARGUMENT_NAME: {
                    "type": "string",
                    "description": CODE_ARGUMENT_DESCRIPTION
                },
            },
            "required": [CODE_ARGUMENT_NAME],
        }

    def description(self) -> str:
        return TOOL_DESCRIPTION

    def system_prompt_fragment(self, tool_name: str) -> str:
        return SYSTEM_PROMPT_FRAGMENT_TEMPLATE.format(tool_name=tool_name, catalog=self.catalog.render_for_prompt())
