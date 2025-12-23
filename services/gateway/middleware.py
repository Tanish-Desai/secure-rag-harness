import os
import logging
import requests
from fastapi import HTTPException

logger = logging.getLogger("gateway.middleware")

# ------------------------------------------------------------------
# Service configuration
# ------------------------------------------------------------------

POLICY_URL = os.getenv("POLICY_URL", "http://policy:8002")
LOGGER_URL = os.getenv("LOGGER_URL", "http://logger:8003")

# ------------------------------------------------------------------
# Policy enforcement
# ------------------------------------------------------------------

def check_policy(query: str, context: list):
    """
    Sends the query and retrieved context to the policy service.
    Raises HTTPException(403) if the request is blocked.
    """
    payload = {
        "query": query,
        "context": [doc.get("content", "") for doc in context],
    }

    try:
        response = requests.post(
            f"{POLICY_URL}/inspect",
            json=payload,
            timeout=1.0,
        )

        if response.status_code == 403:
            logger.warning("Request blocked by policy service.")
            raise HTTPException(
                status_code=403,
                detail="Request blocked by security policy.",
            )

        if response.status_code != 200:
            logger.warning(
                f"Policy service returned unexpected status: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        # Fail-open to avoid blocking experiments if the policy service is unavailable
        logger.warning("Policy service unreachable. Proceeding without enforcement.")

    except requests.exceptions.Timeout:
        logger.warning("Policy service request timed out. Proceeding without enforcement.")

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(f"Unexpected error in policy middleware: {exc}")


# ------------------------------------------------------------------
# Telemetry logging
# ------------------------------------------------------------------

def log_telemetry(metrics: dict):
    """
    Sends telemetry data to the logger service in the background.
    """
    try:
        requests.post(
            f"{LOGGER_URL}/log",
            json=metrics,
            timeout=1.0,
        )
    except Exception as exc:
        # Telemetry failures should not affect request handling
        logger.debug(f"Failed to send telemetry: {exc}")
