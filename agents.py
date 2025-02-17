import numpy as np
from scipy.ndimage import label
from scipy.ndimage import sum_labels
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def neighbors(L):
    v = [(-1,0),(1,0),(0,-1),(0,1)]
    w = np.zeros((L,L),dtype="object")
    for n in range(L):
        for m in range(L):
            w[n,m] = [x for x in v if n+x[0] >= 0 and n+x[0]<L and m+x[1]>=0 and m+x[1]<L]
    return w

if __name__ == "__main__":
    T = 1000 # max number of time steps
    L = 128 # lattice size
    p = 0.465 # occupation probability
    a = (np.random.rand(L,L)<p).astype("int") # innitial populated lattice
    g = neighbors(L) #list of neighbors for each site

    # animation, adjust interval for increasing the speed
    fig, ax = plt.subplots(figsize=(5,5))
    ax.axis('off')
    ims,im = [],ax.imshow(a,animated=True)
    ims.append([im])
    x = np.array([i for i in range(L)])
    y = np.array([j for j in range(L)])
    for t in range(T):
        np.random.shuffle(x)
        np.random.shuffle(y)
        flag = True
        for n in range(L):
            for m in range(L):
                if a[x[n],y[m]] == 1:
                    w = [a[x[n]+c[0],y[m]+c[1]] for c in g[x[n],y[m]]]
                    if np.sum(w) < 2:
                        u = [g[x[n],y[m]][i] for i in range(len(w)) if w[i] == 0]
                        if len(u) > 0:
                            (q,r) = u[np.random.randint(len(u))]
                            a[x[n]+q,y[m]+r] = a[x[n],y[m]]
                            a[x[n],y[m]] = 0
                            flag = False
        im = ax.imshow(a,animated=True)
        ims.append([im])
        if flag:
            break
    ani = animation.ArtistAnimation(fig,ims,interval=100,blit=True,repeat=False)
    plt.show()
    # show clusters
    fig = plt.figure(figsize=(5,5))
    w,n = label(a)
    area = sum_labels(a,w,index=np.arange(n+1)).astype("int")
    plt.imshow(np.sqrt(area[w]),interpolation='nearest')
    plt.axis("off")    
    plt.tight_layout()
    plt.show()

