"""
List of all exceptions found in the program
"""


class InvalidModeException(Exception):
    """Exception raised for invalid mode for video parameters

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="the mode isn't valid - pass a valid image/video path or a webcam stream"):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
