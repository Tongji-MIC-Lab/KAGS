from models.updown import UpDown
from models.xlan_fc_group import XLAN_fc_group

__factory = {
    'XLAN_fc_group': XLAN_fc_group
}
#__factory = {
#    'UpDown': UpDown,
#    'XLAN_bu': XLAN_bu,
#    'XTransformer': XTransformer,
#    'XLAN_fc_group': XLAN_fc_group,
#    'XLAN': XLAN
#}
def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)