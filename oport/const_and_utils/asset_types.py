# asset_types.py
# Utilities for reading and generating asset type mappings.

from .constants import METADATA_ASSET_TYPES, METADATA_ASSET_TYPES_HEADER_FILE  # noqa: F401 (re-exported)


def create_asset_type_map(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    current_category = None
    static_map = {}

    for line in lines:
        line = line.strip()
        if line.startswith("A "):
            current_category = line[2:]
        elif line.startswith("× "):
            key = line[2:]
            if current_category:
                static_map[key] = current_category

    return static_map


def generate_python_file(static_map, output_file):
    with open(output_file, "w") as file:
        for key in static_map.keys():
            variable_name = "ASSET_TYPE_" + key.upper().replace(" ", "_").replace(
                ".", ""
            ).replace("&", "_AND_").replace("-", "_").replace("/", "_").replace(",", "")
            file.write(f"{variable_name} = '{key}'\n")
