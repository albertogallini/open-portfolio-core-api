from typing import List, Dict, Any
import threading

thread_local = threading.local()


class ErrorCollector:
    def __init__(self):
        self.errors: List[Dict[str, str]] = []
        self.status: str = "SUCCESS"

    def record_error(self, operation_name: str, error_details: str):
        self.status = "FAILED"
        self.errors.append({"operation": operation_name, "message": error_details})

    def get_errors(self, result: Any = None):
        if self.status == "FAILED":
            return {
                "status": self.status,
                "message": "API request failed.",
                "errors": self.errors,
            }
        return None

    def pretty_print_errors(self) -> str:
        """
        Returns the collected errors as a clear, formatted, and
        human-readable string, using the raw error message.
        """
        output = ""

        if self.status != "FAILED":
            output += "Collector status is SUCCESS. No errors to display.\n"
            return output

        # 1. Print main status and overall message
        status = self.status
        message = "API request failed."  # Consistent with the standard response

        output += "\n" + "=" * 80 + "\n"
        output += f"| API Status: {status}\n"
        output += f"| Overall Message: {message}\n"
        output += "=" * 80 + "\n"

        # 2. Iterate and print detailed errors
        errors = self.errors
        output += f"Total Detailed Errors: {len(errors)}\n\n"

        for i, error in enumerate(errors):
            operation = error.get("operation", "N/A")
            detail_message = error.get("message", "No detail message.")

            output += f"--- ERROR {i + 1} / OPERATION: {operation} ---\n"

            # Print the raw error message without making assumptions about its structure
            output += f"  Message: {detail_message.strip()}\n"

            output += "-" * 25 + "\n"
        output += "=" * 80 + "\n"

        return output


def get_collector() -> ErrorCollector:
    """Retrieves the collector for the current thread, or creates one if it doesn't exist."""
    if not hasattr(thread_local, "collector"):
        thread_local.collector = ErrorCollector()
    return thread_local.collector


def clear_collector():
    """Cleans up the collector after the request is finished."""
    if hasattr(thread_local, "collector"):
        del thread_local.collector
