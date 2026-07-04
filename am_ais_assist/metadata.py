"""
This module provides module metadata at runtime.
"""

from importlib.metadata import metadata as meta

import pydantic


class MetadataModel(pydantic.BaseModel):
    """Metadata about this module."""

    name: str = pydantic.Field(..., description="The name of this module.")
    description: str = pydantic.Field(..., description="A description of this module.")
    version: str = pydantic.Field(..., description="The version of this module.")
    author: str = pydantic.Field(..., description="The author of this module.")


# get metadata as dict using json property to allow 'get' calls
package_metadata = meta(__package__).json
metadata = MetadataModel(
    name=package_metadata.get("name", "").replace("-", "_"),  # type: ignore
    description=package_metadata.get("summary"),
    version=package_metadata.get("version"),
    author=package_metadata.get("author"),
)
