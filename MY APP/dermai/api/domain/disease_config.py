
from pathlib import Path
from typing import Optional
import yaml


class DiseaseInfo:
    
    def __init__(self, class_id: int, name: str, icd10: str,
                 learn_more_url: str, description: str) -> None:
        self.class_id = class_id
        self.name = name
        self.icd10 = icd10
        self.learn_more_url = learn_more_url
        self.description = description.strip()

    def __repr__(self) -> str:
        return f"<DiseaseInfo id={self.class_id} name={self.name}>"


class DiseaseConfig:
    
    _CONFIG_PATH = Path(__file__).parent.parent.parent / "ml" / "class_config.yaml"

    def __init__(self, config_path: Optional[Path] = None) -> None:
        path = config_path or self._CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._diseases: dict[int, DiseaseInfo] = {
            d["class_id"]: DiseaseInfo(**d)
            for d in raw["diseases"]
        }

    def get_by_index(self, index: int) -> Optional[DiseaseInfo]:
       
        return self._diseases.get(index)

    def all_diseases(self) -> list[DiseaseInfo]:
        return list(self._diseases.values())

    def count(self) -> int:
        return len(self._diseases)
