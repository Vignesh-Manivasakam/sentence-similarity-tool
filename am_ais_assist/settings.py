"""
Configure the module from environment variables.
"""

import typing

import pydantic_settings


class Base(pydantic_settings.BaseSettings):
    """Settings for this module"""

    ROOT_PATH: str = ""
    LOG_LEVEL: typing.Literal[
        "CRITICAL",
        "FATAL",
        "ERROR",
        "WARN",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    ] = "INFO"
    JSON_LOGS: bool = False

    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Base()
