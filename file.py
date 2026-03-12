from copy import deepcopy
from pathlib import Path
from typing import Any

from agentcore.base.data.base_file import BaseFileNode
from agentcore.base.data.utils import IMG_FILE_TYPES, TEXT_FILE_TYPES, parallel_load_data, parse_text_file_to_data
from agentcore.io import BoolInput, FileInput, IntInput, Output
from agentcore.schema.data import Data
from agentcore.services.deps import get_storage_service

_BINARY_FILE_TYPES = [
    *IMG_FILE_TYPES,
    "tiff",
    "bmp",
    "webp",
    "pptx",
    "xlsx",
]


class File(BaseFileNode):

    display_name = "Knowledge Base"
    description = "Select one or more knowledge bases and load their files."
    icon = "file-text"
    name = "File"

    VALID_EXTENSIONS = [*TEXT_FILE_TYPES, *_BINARY_FILE_TYPES]

    _base_inputs = deepcopy(BaseFileNode._base_inputs)

    for input_item in _base_inputs:
        if isinstance(input_item, FileInput) and input_item.name == "path":
            input_item.display_name = "Knowledge Bases"
            input_item.real_time_refresh = True
            break

    inputs = [
        *_base_inputs,
        BoolInput(
            name="use_multithreading",
            display_name="Use Multithreading",
            advanced=True,
            value=True,
            info="Set 'Processing Concurrency' greater than 1 to enable multithreading.",
        ),
        IntInput(
            name="concurrency_multithreading",
            display_name="Processing Concurrency",
            advanced=True,
            info="When multiple files are being processed, the number of files to process concurrently.",
            value=1,
        ),
    ]

    outputs = [
        Output(display_name="Raw Content", name="message", method="load_files_message"),
    ]

    @staticmethod
    def _as_path_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item]
        if isinstance(value, dict):
            file_path_value = value.get("file_path")
            if isinstance(file_path_value, str):
                return [file_path_value] if file_path_value else []
            if isinstance(file_path_value, list):
                return [item for item in file_path_value if isinstance(item, str) and item]
        return []

    @staticmethod
    def _looks_like_path(path_value: str) -> bool:
        # Stored file paths are like "<user_id>/<kb_name>/<file_name.ext>".
        return "/" in path_value or "\\" in path_value

    def _has_selectable_content(self, path_value: str) -> bool:
        """Return True when a path points to at least one processable file."""
        if not path_value:
            return False

        candidate_paths = [Path(self.resolve_path(path_value))]
        storage_root = Path(str(get_storage_service().data_dir))
        candidate_paths.append(storage_root / path_value)
        if "/" in path_value:
            try:
                candidate_paths.append(Path(self.get_full_path(path_value)))
            except Exception:
                pass
        user_id = getattr(self, "user_id", None)
        if user_id and not path_value.startswith(f"{user_id}/"):
            candidate_paths.append(storage_root / str(user_id) / path_value)

        supported_extensions = set(self.valid_extensions) | set(self.SUPPORTED_BUNDLE_EXTENSIONS)

        resolved_path = next((path for path in candidate_paths if path.exists()), None)
        if resolved_path is None:
            return False

        if resolved_path.is_file():
            suffix = resolved_path.suffix[1:].lower()
            return suffix in supported_extensions

        if resolved_path.is_dir():
            return any(
                candidate.is_file() and candidate.suffix[1:].lower() in supported_extensions
                for candidate in resolved_path.rglob("*")
            )

        return False

    def _filter_selectable_paths(self, path_values: list[str]) -> list[str]:
        return [path_value for path_value in path_values if self._has_selectable_content(path_value)]

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically show only the relevant output based on the number of files processed."""
        if field_name == "path":
            path_template = frontend_node.get("template", {}).get("path")
            template_file_paths = []
            if isinstance(path_template, dict):
                template_file_paths = self._as_path_list(path_template.get("file_path"))

            incoming_paths = self._as_path_list(field_value)
            # Realtime update sends KB names in field_value; actual storage paths are in template.path.file_path.
            if template_file_paths and (
                not incoming_paths or all(not self._looks_like_path(path) for path in incoming_paths)
            ):
                selected_paths = template_file_paths
            else:
                selected_paths = incoming_paths

            filtered_paths = self._filter_selectable_paths(selected_paths)
            invalid_paths = [path for path in selected_paths if path not in filtered_paths]

            if invalid_paths:
                self.log(
                    "Ignoring unresolved/unsupported knowledge base selections: "
                    + ", ".join(invalid_paths)
                )
                # Keep selection stable; avoid UI popup/revert for legacy path tokens.
                if not filtered_paths and selected_paths:
                    filtered_paths = selected_paths

            if isinstance(path_template, dict):
                path_template["file_path"] = filtered_paths

            field_value = filtered_paths

            # Add outputs based on the number of files in the path
            if len(field_value) == 0:
                frontend_node["outputs"] = []
                return frontend_node

            frontend_node["outputs"] = []

            if len(field_value) == 1:
                # We need to check if the file is structured content
                file_path = frontend_node["template"]["path"]["file_path"][0]
                if file_path.endswith((".csv", ".xlsx", ".parquet")):
                    frontend_node["outputs"].append(
                        Output(display_name="Structured Content", name="dataframe", method="load_files_structured"),
                    )
                elif file_path.endswith(".json"):
                    frontend_node["outputs"].append(
                        Output(display_name="Structured Content", name="json", method="load_files_json"),
                    )

            # Always include path output so OCR and other downstream components can connect
            frontend_node["outputs"].append(
                Output(display_name="Knowledge Base", name="path", method="load_files_path"),
            )

            if len(field_value) > 1:
                # For multiple files, also show the combined files output
                frontend_node["outputs"].append(
                    Output(display_name="Knowledge Bases", name="dataframe", method="load_files"),
                )

        return frontend_node

    def process_files(self, file_list: list[BaseFileNode.BaseFile]) -> list[BaseFileNode.BaseFile]:
        """Processes files either sequentially or in parallel, depending on concurrency settings.

        Args:
            file_list (list[BaseFileNode.BaseFile]): List of files to process.

        Returns:
            list[BaseFileNode.BaseFile]: Updated list of files with merged data.
        """

        def process_file(file_path: str, *, silent_errors: bool = False) -> Data | None:
            """Processes a single file and returns its Data object."""
            ext = Path(file_path).suffix.lstrip(".").lower()

            # Binary files (images, pptx, xlsx, etc.) can't be parsed as text.
            # Return a Data object with the file path so downstream components
            # (e.g. Document OCR Extractor) can consume them.
            if ext in _BINARY_FILE_TYPES:
                return Data(
                    text=file_path,
                    data={"file_path": file_path, "file_type": ext},
                )

            try:
                return parse_text_file_to_data(file_path, silent_errors=silent_errors)
            except FileNotFoundError as e:
                msg = f"File not found: {file_path}. Error: {e}"
                self.log(msg)
                if not silent_errors:
                    raise
                return None
            except Exception as e:
                msg = f"Unexpected error processing {file_path}: {e}"
                self.log(msg)
                if not silent_errors:
                    raise
                return None

        if not file_list:
            msg = "No files to process."
            raise ValueError(msg)

        concurrency = 1 if not self.use_multithreading else max(1, self.concurrency_multithreading)
        file_count = len(file_list)

        parallel_processing_threshold = 2
        if concurrency < parallel_processing_threshold or file_count < parallel_processing_threshold:
            if file_count > 1:
                self.log(f"Processing {file_count} files sequentially.")
            processed_data = [process_file(str(file.path), silent_errors=self.silent_errors) for file in file_list]
        else:
            self.log(f"Starting parallel processing of {file_count} files with concurrency: {concurrency}.")
            file_paths = [str(file.path) for file in file_list]
            processed_data = parallel_load_data(
                file_paths,
                silent_errors=self.silent_errors,
                load_function=process_file,
                max_concurrency=concurrency,
            )

        # Use rollup_basefile_data to merge processed data with BaseFile objects
        return self.rollup_data(file_list, processed_data)
