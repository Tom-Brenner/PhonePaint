"""Fixed MAPS phone inventory (61 classes)."""

PHONES = (
    "h# q eh dx iy r ey ix tcl sh ow z s hh aw m t er l w aa hv ae dcl y "
    "axr d kcl k ux ng gcl g ao epi ih p ay v n f jh ax en oy dh pcl ah "
    "bcl el zh uw pau b uh th ax-h em ch nx eng"
).split()

N_FEATURES = 39
N_CLASSES = len(PHONES)

num2phn = {i: p for i, p in enumerate(PHONES)}
phn2num = {p: i for i, p in enumerate(PHONES)}
phn2num["sil"] = phn2num["h#"]
