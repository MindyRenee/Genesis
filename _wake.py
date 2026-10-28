import sys
import time

sys.path.insert(0, '/home/mindy/Public/genesis')
from genesis_client.client import GenesisClient

c = GenesisClient('/home/mindy/.local/share/genesis/genesis.sock')
c.connect()
def snap(tag):
    s = c.get_state()
    ch = s.chemicals
    print(
        f"{tag}: phase={s.phase_name} arousal={s.arousal:.3f} "
        f"cort={ch['cortisol']:.3f} orex={ch['orexin']:.3f} "
        f"htm={ch['histamine']:.3f} adn={ch['adenosine']:.3f}",
        flush=True,
    )
    return s
snap('start')
flipped = False
for i in range(400):                    # up to ~27 min at 4s cadence
    c.stress(0.04)
    c.neuro_impulse(15, 0.10)
    c.neuro_impulse(9, 0.08)
    time.sleep(4.0)
    s = c.get_state()
    if i % 7 == 0:
        snap(f"t+{(i+1)*4:.0f}s")
    if s.phase_name not in ('nrem', 'rem'):
        print('--- WAKE SIDE ---', flush=True)
        snap('flip')
        flipped = True
        break
if not flipped:
    print('--- still nrem after sustained drive ---', flush=True)
for i in range(20):
    time.sleep(3.0)
    snap(f"settle+{i*3:.0f}s")
