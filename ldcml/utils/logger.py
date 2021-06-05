import logging
import csv
import io
from typing import List


class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_NONNUMERIC)

    def format(self, record: logging.LogRecord):
        self.writer.writerow([record.msg, *record.args])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


class CsvLogger():
    def __init__(self, columns: List[str], filename: str):
        self.__logger = logging.Logger(filename)
        self.__logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")
        handler.setFormatter(CsvFormatter())
        self.__logger.addHandler(handler)
        self.__logger.info(*columns)

    def info(self, *args):
        self.__logger.info(*args)
