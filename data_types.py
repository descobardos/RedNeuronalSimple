from typing import Sequence, TypeVar, Iterable, Tuple, cast

# informacion en PEP 696 https://peps.python.org/pep-0696/#using-another-typevarlike-as-default

def slice_data():
    pixel_slice = TypeVar('pixel_slice', bound=Sequence) 
    return pixel_slice

