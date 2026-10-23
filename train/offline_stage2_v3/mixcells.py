# [v3-ic2] demo-cell-mix initial conditions shared by the ic2 gate and the std smoke (new file).
import numpy as np
DEMO_CELLS=[(5,2.,240)]+[(p,y,80) for p in (10,20,50) for y in (5.,10.,15.)]+[(40,10.,80),(40,30.,80),(60,20.,80),(60,30.,80),(60,45.,79),(80,10.,80),(80,20.,80),(80,30.,78),(80,45.,78)]
def demo_pose(rng,p,y):
    dx,dy=rng.uniform(-p,p,2);yaw=rng.uniform(-y,y);u=rng.random()
    if p<=.005:return float(dx),float(dy),float(yaw)
    if u<.08:dx=p*rng.choice([-1,1]);dy=p*rng.choice([-1,1]);yaw=y*rng.choice([-1,1])
    elif u<.32:
        dim=int(rng.integers(3));sg=rng.choice([-1,1])
        if dim==0:dx=sg*p
        elif dim==1:dy=sg*p
        else:yaw=sg*y
    return float(dx),float(dy),float(yaw)
def alloc(n):
    w=np.array([c[2] for c in DEMO_CELLS],float);q=n*w/w.sum();c=np.floor(q).astype(int)
    for i in np.argsort(-(q-c))[:n-c.sum()]:c[i]+=1
    return c
def cell_name(p,y):return f'p{p}_y{int(y)}'
def stratum(p,y):return 'nominal' if p<=5 else ('expanded' if p in (10,20,50) else 'wide')
def mix_episodes(seed0,n):
    """Exact (largest-remainder) demo cell mix, in-cell poses = demo collectors' pose_for (corner/boundary/uniform)."""
    E=[];s=seed0
    for (p,y,_),k in zip(DEMO_CELLS,alloc(n)):
        for _ in range(k):
            dx,dy,yw=demo_pose(np.random.default_rng(s+5000000),p/1000,y)
            E.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell=cell_name(p,y),stratum=stratum(p,y),pos_half_mm=p,yaw_half_deg=y));s+=1
    return E
CELL_ORDER=[cell_name(p,y) for p,y,_ in DEMO_CELLS]
