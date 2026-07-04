"""
Some utility functions for use in Modanalit pages.
"""

import os

import am_ais_assist


def get_version() -> str:
    """Returns this module's version."""
    version: str = am_ais_assist.__version__  # this is the default

    image_tag = os.getenv("MODULE_IMAGE_TAG", default=None)
    # For `main` and `latest`, housekeeping makes sure the version number is correct.
    # For all other tags, show the tag.
    if image_tag and image_tag not in ("main", "latest"):
        version = image_tag

    local_run = os.getenv("DEVELOPMENT_MODE_LOCAL", default="True")
    if local_run == "True":
        version = "local"

    return version
