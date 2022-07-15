class Error(Exception):
    pass


class LongitudeError(Error):
    """Exception raised when the longitude coordinate of a Dataset/DataArray
    doesn't span the globe or when it is irregularly spaced"""

    def __init__(self, message):
        self.message = message


class CoordinateError(Error):
    """Exception raised when the input Dataset/DataArray doesn't have the
    appropriate coordinate(s)."""

    def __init__(self, message):
        self.message = message


class AttrError(Error):
    """Exception raised when the input Dataset/DataArray doesn't have the
    appropriate attribute(s)."""

    def __init__(self, message):
        self.message = message


class UnitError(Error):
    """Exception raised when the input DataArray doesn't have the
    appropriate units string for the given variable."""

    def __init__(self, message):
        self.message = message
