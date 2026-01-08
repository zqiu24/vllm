# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from typing import Optional

import msgspec


class OFTRequest(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """
    Request for a OFT adapter.

    Note that this class should be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized OFT adapters.

    oft_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """
    oft_name: str
    oft_int_id: int
    oft_path: str = ""
    oft_local_path: Optional[str] = msgspec.field(default=None)
    long_oft_max_len: Optional[int] = None
    base_model_name: Optional[str] = msgspec.field(default=None)
    tensorizer_config_dict: Optional[dict] = None

    def __post_init__(self):
        if self.oft_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.oft_int_id}")
        if self.oft_local_path:
            warnings.warn(
                "The 'oft_local_path' attribute is deprecated "
                "and will be removed in a future version. "
                "Please use 'oft_path' instead.",
                DeprecationWarning,
                stacklevel=2)
            if not self.oft_path:
                self.oft_path = self.oft_local_path or ""

        # Ensure oft_path is not empty
        assert self.oft_path, "oft_path cannot be empty"

    @property
    def adapter_id(self):
        return self.oft_int_id

    @property
    def name(self):
        return self.oft_name

    @property
    def path(self):
        return self.oft_path

    @property
    def local_path(self):
        warnings.warn(
            "The 'local_path' attribute is deprecated "
            "and will be removed in a future version. "
            "Please use 'path' instead.",
            DeprecationWarning,
            stacklevel=2)
        return self.oft_path

    @local_path.setter
    def local_path(self, value):
        warnings.warn(
            "The 'local_path' attribute is deprecated "
            "and will be removed in a future version. "
            "Please use 'path' instead.",
            DeprecationWarning,
            stacklevel=2)
        self.oft_path = value

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare OFTRequest
        instances based on oft_name. This allows for identification
        and comparison oft adapter across engines.
        """
        return isinstance(value,
                          self.__class__) and self.oft_name == value.oft_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash OFTRequest instances
        based on oft_name. This ensures that OFTRequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.oft_name)
