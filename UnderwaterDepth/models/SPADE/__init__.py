from .SPADE_v1 import SPADE

all_versions = {
    "v1": SPADE
}

get_version = lambda v : all_versions[v]