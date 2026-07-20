from oport.const_and_utils import *

static_map = create_asset_type_map(METADATA_ASSET_TYPES)
generate_python_file(static_map, METADATA_ASSET_TYPES_HEADER_FILE)
