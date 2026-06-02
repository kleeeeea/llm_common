from dataclasses import dataclass



@dataclass(frozen=True)
class EvaluationInstance(object):
    value:str

    def from_dict(self, data: dict) -> None:
        pass
