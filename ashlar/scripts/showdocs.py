import inspect
import itertools
import attr

from ashlar.metadata import Tile, TileSet, ImageReader
from ashlar.geometry import Vector, Rectangle
from ashlar.align import TileAlignment
from ashlar.plot import TileSetPlotter
#from ashlar.bioformats import BioformatsReader

kind_order = {
    'property': 1,
    'method': 2,
    'class method': 3,
}

classes = [
    Vector, Rectangle, Tile, TileSet, ImageReader, TileAlignment,
    TileSetPlotter,
    #BioformatsReader
]

for cls in classes:
    module = inspect.getmodule(cls)
    csig = str(inspect.signature(cls.__init__))
    csig = csig.replace('self, ', '').replace(' -> None', '')
    print(
        f"\x1b[1;36m{cls.__name__}\x1b[0;36m{csig}\x1b[0m"
        f"  \x1b[2m<{module.__name__}>\x1b[0m"
    )
    cdoc = inspect.getdoc(cls)
    if cdoc:
        first_line = cdoc.split('\n', 1)[0]
        print(f"    {first_line}")
    base_attrs = [
        a for a in inspect.classify_class_attrs(cls)
        if not a.name.startswith('__') and a.kind != 'data'
    ]
    base_attrs = sorted(base_attrs, key=lambda a: (kind_order[a.kind], a.name))
    attr_attrs = [
        inspect.Attribute(a.name, 'attr', cls, a) for a in attr.fields(cls)
    ]
    attrs = attr_attrs + base_attrs
    for kind, grouper in itertools.groupby(attrs, lambda a: a.kind):
        for name, kind, def_class, obj in grouper:
            if kind == 'class method':
                obj = obj.__func__
                name = f"{cls.__name__}.{name}"
            sig = ''
            if kind in ('method', 'class method'):
                sig = str(inspect.signature(obj))
                if kind == 'class method':
                    sig = sig.replace('(cls, ', '(')
            print(f"    \u2022 \x1b[1;32m{name}\x1b[0;32m{sig}\x1b[0m", end="")
            doc = inspect.getdoc(obj)
            if kind == 'attr':
                doc = obj.metadata.get('doc')
            if doc:
                first_line = doc.split('\n', 1)[0]
                print(f" : {first_line}", end="")
            print()
    print()
