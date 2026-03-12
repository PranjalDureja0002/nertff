import json
from typing import Any

from agentcore.logging import logger
from agentcore.inputs.inputs import (
    BoolInput,
    HandleInput,
    MessageTextInput,
    MultilineInput,
    TableInput,
)
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.custom.custom_node.node import Node

class SmartRouterComponent(Node):
    """Smart Router component that uses semantic understanding to route inputs.

    Instead of relying on rigid, rule-based conditions, Smart Router uses an LLM
    to semantically understand the input content and decide which route is most appropriate.
    This enables intelligent routing for intent-based workflows, multi-agent systems,
    and dynamic task delegation.
    """

    display_name = "Smart Router"
    description = "Routes an input message using LLM-based categorization."
    icon = "route"
    name = "SmartRouter"

    inputs = [
        HandleInput(
            name="router_llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            required=True,
            info="The LLM that will analyze the input and decide which route to take.",
        ),
        HandleInput(
            name="input_data",
            display_name="Input",
            input_types=["Message", "Data"],
            required=True,
            info="The input message or data to route. Can be from an Agent or any other component.",
        ),
        TableInput(
            name="routes",
            display_name="Routes",
            info="Define available routes with descriptions. The LLM will choose based on semantic understanding.",
            table_schema=[
                {
                    "name": "route_name",
                    "display_name": "Route Name",
                    "type": "str",
                    "description": "Unique identifier for this route (shows as output handle)",
                },
                {
                    "name": "description",
                    "display_name": "Description",
                    "type": "str",
                    "description": "Describe when this route should be selected",
                },
            ],
            value=[
                {"route_name": "Route 1", "description": "First route - describe when to use this route"},
                {"route_name": "Route 2", "description": "Second route - describe when to use this route"},
                {"route_name": "Route 3", "description": "Third route - describe when to use this route"},
            ],
            real_time_refresh=True,
        ),
        MultilineInput(
            name="system_context",
            display_name="Routing Context",
            info="Additional context to help the LLM understand the routing domain (optional).",
            value="",
            advanced=True,
        ),
        BoolInput(
            name="include_reasoning",
            display_name="Include Reasoning",
            value=True,
            info="Include the LLM's reasoning in the routing info output.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Route 1", name="Route 1", method="route_output", group_outputs=True, types=["Message", "Data"]),
        Output(display_name="Route 2", name="Route 2", method="route_output", group_outputs=True, types=["Message", "Data"]),
        Output(display_name="Route 3", name="Route 3", method="route_output", group_outputs=True, types=["Message", "Data"]),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_route: str | None = None
        self._routing_reasoning: str = ""
        self._route_evaluated: bool = False

    def _extract_input_text(self) -> str:
        """Extract text content from the input for semantic analysis."""
        input_val = self.input_data

        if isinstance(input_val, Message):
            return input_val.text or ""

        if isinstance(input_val, Data):
            if isinstance(input_val.data, dict):
                return json.dumps(input_val.data, indent=2)
            return str(input_val.data)

        if isinstance(input_val, dict):
            return json.dumps(input_val, indent=2)

        return str(input_val)

    def _get_route_names(self) -> list[str]:
        """Get list of defined route names."""
        routes = []
        for route in self.routes or []:
            if isinstance(route, dict) and route.get("route_name"):
                routes.append(route["route_name"].strip())
        return routes

    def _build_routing_prompt(self, input_text: str) -> str:
        """Build the prompt for the LLM to make a routing decision."""
        route_descriptions = []
        for i, route in enumerate(self.routes or [], 1):
            if isinstance(route, dict):
                name = route.get("route_name", f"route_{i}")
                desc = route.get("description", "No description provided")
                route_descriptions.append(f"- **{name}**: {desc}")

        routes_text = "\n".join(route_descriptions)

        system_context = ""
        if self.system_context and self.system_context.strip():
            system_context = f"\n\n**Domain Context:**\n{self.system_context.strip()}\n"

        prompt = f"""You are an intelligent router that analyzes input and determines the most appropriate route.

**Available Routes:**
{routes_text}
{system_context}
**Input to Route:**
{input_text}

**Instructions:**
1. Analyze the input's intent, content, and purpose
2. Select the SINGLE most appropriate route based on semantic understanding
3. Respond with ONLY a JSON object in this exact format:

{{"selected_route": "route_name_here", "reasoning": "Brief explanation of why this route was chosen"}}

Important: The "selected_route" must be one of the exact route names listed above."""

        return prompt

    def _parse_llm_response(self, response_text: str) -> tuple[str, str]:
        """Parse the LLM response to extract route and reasoning."""
        try:
            clean_response = response_text.strip()
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()

            start_idx = clean_response.find("{")
            end_idx = clean_response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = clean_response[start_idx:end_idx]
                result = json.loads(json_str)
                selected_route = result.get("selected_route", "").strip()
                reasoning = result.get("reasoning", "")
                return selected_route, reasoning
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

        # Fallback: try to find route name in response
        route_names = self._get_route_names()
        response_lower = response_text.lower()
        for route_name in route_names:
            if route_name.lower() in response_lower:
                return route_name, f"Extracted from response: {response_text[:200]}"

        return "", f"Could not parse response: {response_text[:200]}"

    async def _evaluate_route(self) -> str:
        """Use the LLM to semantically evaluate and select the best route."""
        if self._route_evaluated:
            return self._selected_route or ""

        self._route_evaluated = True
        route_names = self._get_route_names()
        default_route = route_names[0] if route_names else ""

        input_text = self._extract_input_text()

        if not input_text.strip():
            self._selected_route = default_route
            self._routing_reasoning = "Empty input - using first route"
            return self._selected_route

        prompt = self._build_routing_prompt(input_text)

        try:
            response = await self.router_llm.ainvoke(prompt)

            if hasattr(response, "content"):
                response_text = response.content
            elif hasattr(response, "text"):
                response_text = response.text
            else:
                response_text = str(response)

            selected_route, reasoning = self._parse_llm_response(response_text)

            if selected_route in route_names:
                self._selected_route = selected_route
                self._routing_reasoning = reasoning
                logger.debug(f"Smart router selected route: {selected_route}")
            else:
                self._selected_route = default_route
                self._routing_reasoning = f"LLM selected '{selected_route}' which is not valid. Using default. Original reasoning: {reasoning}"
                logger.warning(f"Invalid route '{selected_route}' - defaulting to: {default_route}")

        except Exception as e:
            self._selected_route = default_route
            self._routing_reasoning = f"Error during routing: {e!s}. Using first route."
            logger.error(f"Error during routing: {e!s}. Using first route.")

        return self._selected_route

    def _sync_evaluate_route(self) -> str:
        """Synchronous wrapper for route evaluation."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._evaluate_route())
                    return future.result()
            else:
                return loop.run_until_complete(self._evaluate_route())
        except RuntimeError:
            return asyncio.run(self._evaluate_route())

    def route_output(self) -> Message | Data:
        """Generic method to handle route output - called for each dynamic output."""
        selected = self._sync_evaluate_route()

        # Get which output is currently being processed
        current_output_name = self._current_output

        if selected == current_output_name:
            self.status = f"Routed to: {current_output_name}"
            if self.include_reasoning and self._routing_reasoning:
                self.status += f"\nReasoning: {self._routing_reasoning}"

            # Stop all other routes
            route_names = self._get_route_names()
            for route in route_names:
                if route != current_output_name:
                    self.stop(route)
            return self.input_data

        self.stop(current_output_name)
        return Message(text="")

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically update outputs based on routes table."""
        if field_name == "routes" and field_value:
            # Build new outputs from routes table
            frontend_node["outputs"] = []
            for route in field_value:
                if isinstance(route, dict) and route.get("route_name"):
                    route_name = route["route_name"].strip()
                    if route_name:
                        frontend_node["outputs"].append(
                            Output(
                                display_name=route_name,
                                name=route_name,
                                method="route_output",
                                group_outputs=True,
                                types=["Message", "Data"],
                            )
                        )

        return frontend_node

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None) -> dict:
        """Update build config based on routes table."""
        if field_name == "routes" and field_value:
            route_names = []
            for route in field_value:
                if isinstance(route, dict) and route.get("route_name"):
                    route_names.append(route["route_name"].strip())
            build_config["_configured_routes"] = route_names

        return build_config


