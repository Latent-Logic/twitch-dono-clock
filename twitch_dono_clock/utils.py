from abc import ABCMeta


class Singleton(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        try:
            return cls._instances[cls]
        except KeyError:
            cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
