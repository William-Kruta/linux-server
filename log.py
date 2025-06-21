import datetime


class Log:
    """
    A simple class to handle logging messages to a text file.
    """

    def __init__(self, filename="log.txt"):
        """
        Initializes the Log class.

        Args:
            filename (str): The name of the file to save logs to.
                            Defaults to "log.txt".
        """
        self.filename = filename
        # The 'with open...' statement here ensures the file is created
        # if it doesn't exist when the Log object is instantiated.
        try:
            with open(self.filename, "a"):
                pass
        except IOError as e:
            print(
                f"Error: Could not create or open log file '{self.filename}'.\nDetails: {e}"
            )

    def add(self, message):
        """
        Adds a new message to the log file with a timestamp.

        Args:
            message (str): The message to be logged.
        """
        try:
            # 'with' statement ensures the file is properly closed even if errors occur.
            with open(self.filename, "a") as log_file:
                # Get the current time and format it
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Write the timestamp and message to the file
                log_file.write(f"[{timestamp}] - {message}\n")
        except IOError as e:
            print(
                f"Error: Could not write to log file '{self.filename}'.\nDetails: {e}"
            )
