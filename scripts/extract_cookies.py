"""Extract HARVESTED_COOKIES from test_indices_tradingview.py and write them as environment variables.

Usage:
    python tests/extract_cookies.py [--output .env]

This script reads the HARVESTED_COOKIES dict from tests/test_indices_tradingview.py and writes
them as environment variables (or to a .env file) so they can be used by tests.
"""

import re
import sys
import os
from pathlib import Path


def extract_harvested_cookies_from_dict_literal(filepath: str) -> dict | None:
    """Extract HARVESTED_COOKIES from a Python file where it's defined as a dict literal.

    Returns a dict of cookie_name -> value, or None if not found.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Match the HARVESTED_COOKIES dict definition (handles multi-line dicts)
    pattern = r"HARVESTED_COOKIES\s*=\s*{(.+)}"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None

    cookies_str = match.group(1).strip()

    # Parse the dict manually (handles single-quoted keys/values)
    cookies = {}
    key_pattern = r"'([^']*)'\s*:\s*'([^']*)'"
    for key_match in re.finditer(key_pattern, cookies_str):
        name = key_match.group(1)
        value = key_match.group(2)
        cookies[name] = value

    return cookies


def extract_harvested_cookies_from_function(filepath: str) -> dict | None:
    """Extract HARVESTED_COOKIES from a Python file where it's defined via _load_harvested_cookies().

    Returns a dict of cookie_name -> value, or None if not found.
    """
    import sys
    
    # Add the file's directory to sys.path so we can import it
    dirpath = str(Path(filepath).parent)
    if dirpath not in sys.path:
        sys.path.insert(0, dirpath)
    
    module_name = Path(filepath).stem  # e.g., 'test_indices_tradingview'
    
    try:
        import importlib
        importlib.import_module(module_name)
        mod = sys.modules[module_name]
        
        if hasattr(mod, '_load_harvested_cookies') and callable(mod._load_harvested_cookies):
            return mod._load_harvested_cookies()
    except (ImportError, ModuleNotFoundError):
        pass
    
    return None


def extract_harvested_cookies(filepath: str) -> dict | None:
    """Extract HARVESTED_COOKIES from the given Python file.

    Supports both dict literal and _load_harvested_cookies() function formats.
    Returns a dict of cookie_name -> value, or None if not found.
    """
    cookies = extract_harvested_cookies_from_dict_literal(filepath)
    if cookies:
        return cookies

    cookies = extract_harvested_cookies_from_function(filepath)
    if not cookies:
        print("ERROR: Could not find HARVESTED_COOKIES in", filepath)

    return cookies


def write_env_file(cookies: dict, output_path: str) -> None:
    """Write cookies as environment variables to a .env file."""
    with open(output_path, "w") as f:
        for name, value in cookies.items():
            # Sanitize the cookie name to be a valid env var name
            env_name = f"COOKIE_{name.replace('.', '_').replace('-', '_')}"
            
            # Handle values containing double quotes - use single quotes for the outer wrapper
            if '"' in value and "'" not in value:
                f.write(f"{env_name}='{value}'\n")
            else:
                # Escape single quotes in the value (for shell)
                escaped_value = value.replace("'", "''")
                f.write(f'{env_name}="{escaped_value}"\n')

    print(f"Written {len(cookies)} cookies to {output_path}")


def write_to_env_vars(cookies: dict) -> None:
    """Write cookies directly as environment variables in the current process."""
    for name, value in cookies.items():
        env_name = f"COOKIE_{name.replace('.', '_').replace('-', '_')}"
        os.environ[env_name] = value

    print(f"Wrote {len(cookies)} cookies as environment variables")


def main():
    output_path = ".env"

    # Parse command-line arguments
    args = sys.argv[1:]
    for arg in args:
        if arg == "--output":
            output_path = args[args.index(arg) + 1]

    # Find the test file
    script_dir = Path(__file__).parent.parent  # project root
    test_file = script_dir / "tests" / "test_indices_tradingview.py"

    if not test_file.exists():
        print(f"ERROR: Could not find {test_file}")
        sys.exit(1)

    cookies = extract_harvested_cookies(str(test_file))
    if not cookies:
        sys.exit(1)

    # Write to .env file (default behavior) or as env vars
    if "--as-env-vars" in args:
        write_to_env_vars(cookies)
    else:
        write_env_file(cookies, output_path)


if __name__ == "__main__":
    main()
