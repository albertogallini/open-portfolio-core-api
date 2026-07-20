def get_fields() -> dict:
    fields = {}
    with open("oport/fields.py", "r") as file:
        for line in file:
            if not line.strip() or not line.startswith("FIELD_"):
                continue

            key, value = line.strip().split("=")
            value = value.strip().strip('"').strip("'")
            # Add the key-value pair to the dictionary
            fields[key.strip()] = value
            # print("field : {} -> {}".format(key,value))
    return fields


def get_sources() -> dict:
    srcs = {}
    with open("oport/const_and_utils.py", "r") as file:
        for line in file:
            if line.startswith("CONFIG_SOURCE_"):
                key, value = line.strip().split("=")
                value = value.strip().strip('"').strip("'")
                # Add the key-value pair to the dictionary
                srcs[key.strip()] = value
            else:
                continue
    return srcs


def update_reference_josn_fields_possible_values(fields: dict, file_name) -> None:
    import json

    with open(file_name, "r") as f:
        data = json.load(f)

    acc_fields = [
        value for key, value in fields.items() if key.startswith("FIELD_EVAL_ACC_")
    ]
    data["paths"]["/oport/portfolio_totals"]["parameters"]["oport"][2][
        "choices"
    ] = acc_fields
    data["paths"]["/oport/portfolio_totals"]["parameters"]["oport"][2]["type"] = (
        "Literal" + str(acc_fields)
    )

    with open("reference.json", "w") as f:
        json.dump(data, f, indent=4)


def update_reference_josn_src_possible_values(srcs: dict, file_name) -> None:
    import json

    with open(file_name, "r") as f:
        data = json.load(f)

    srcs_values = [value for key, value in srcs.items()]
    data["paths"]["/oport/impute_positions"]["parameters"]["oport"][4][
        "choices"
    ] = srcs_values
    data["paths"]["/oport/impute_positions"]["parameters"]["oport"][4]["type"] = (
        "Literal" + str(srcs_values)
    )
    data["paths"]["/oport/holdings"]["parameters"]["oport"][8]["choices"] = srcs_values
    data["paths"]["/oport/holdings"]["parameters"]["oport"][8]["type"] = (
        "Literal" + str(srcs_values)
    )
    data["paths"]["/oport/portfolio_totals"]["parameters"]["oport"][6][
        "choices"
    ] = srcs_values
    data["paths"]["/oport/portfolio_totals"]["parameters"]["oport"][6]["type"] = (
        "Literal" + str(srcs_values)
    )

    with open("reference.json", "w") as f:
        json.dump(data, f, indent=4)


import openbb

openbb.build()

import os

print("Current working directory: {}".format(os.getcwd()))

# update_reference_josn_fields_possible_values(get_fields(),"oport/assets/reference.json")
# update_reference_josn_src_possible_values(get_sources(),"reference.json")

import site

# Adjust paths based on your environment
print("Site package dir : {}".format(site.getsitepackages()[0]))
asset_dir = site.getsitepackages()[0] + "/openbb/assets/"

import shutil

shutil.copy(f"./reference.json", f"{asset_dir}/reference.json")

print("Copied reference.json to OpenBB assets folder.")


exit()
