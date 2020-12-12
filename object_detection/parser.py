import json
from pathlib import Path
from typing import List, Union
from icevision.all import parsers, IDMap, ClassMap, BBox


class ArtifactParser(parsers.FasterRCNN, parsers.FilepathMixin, parsers.SizeMixin):
    def __init__(self, df):
        self.df = df
        self.imageid_map = IDMap()
        self.class_map = ClassMap(["Port", "Necklace", "Endo", "ICD"])

    def __iter__(self):
        yield from self.df.itertuples()

    def __len__(self):
        return len(self.df)

    def imageid(self, o) -> int:
        return self.imageid_map[o.image_id]

    def filepath(self, o) -> Union[str, Path]:
        return o.img_path

    def height(self, o) -> int:
        return o.height

    def width(self, o) -> int:
        return o.width

    def labels(self, o) -> List[int]:
        return [self.class_map.get_name(o.label)]

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xyxy(*json.loads(o.bbox))]
