from abc import ABC, abstractmethod

class ILogger(ABC):
    def __init__(self, key_filter):
        self.key_filter = set(key_filter) if key_filter is not None else None

    @abstractmethod
    def _log_dict(self, dictionary: dict):
        raise NotImplementedError()

    def log_dict(self, dictionary: dict):
        # filtering input dict
        if self.key_filter is None:
            filtered = dictionary
        else:
            filtered = {}
            for k, v in dictionary.items():
                if k in self.key_filter:
                    filtered[k] = v
        # child class is logging
        self._log_dict(filtered)


class ConsoleLogger(ILogger):
    def __init__(self, separator=' ', key_filter=None):
        super().__init__(key_filter)
        self.separator = separator

    @staticmethod
    def __val_transform(val):
        if isinstance(val, float):
            return f'{val:.7f}'
        return str(val)

    def _log_dict(self, dictionary: dict):
        message = self.separator.join(f'{k}:\t{self.__val_transform(v)}' for k, v in dictionary.items())
        print(message)


class MemoryLogger(ILogger):
    def __init__(self, key_filter=None):
        super().__init__(key_filter)
        self.log = {}

    def _log_dict(self, dictionary: dict):
        for k, v in dictionary.items():
            if k not in self.log.keys():
                self.log[k] = []
            self.log[k].append(v)
