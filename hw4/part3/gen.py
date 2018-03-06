import numpy as np
from sklearn.neighbors import NearestNeighbors


def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


def get_eigenvalues(data,mode='default',SAMPLE=0,NEIGHBOR=0):
    if SAMPLE<1:
        SAMPLE = int(np.size(data,0)/500) # sample some points to estimate
    if NEIGHBOR<1:
        NEIGHBOR = int(np.size(data,0)/100) # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for i,idx in enumerate(randidx):
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,:]]
        nbrs -= nbrs.mean(axis=0)
        u, s, v = np.linalg.svd(nbrs.T @ nbrs)
        s /= s.max()
        sing_vals.append(s)
    
    sing_vals = np.array(sing_vals)
    if mode=='default':
        sing_vals = sing_vals.mean(axis=0)
    return sing_vals


# generate some data for training
if __name__ == '__main__':
    X = np.zeros((300,100))
    y = np.zeros((300,))
    for i in range(60):
        print('round {}...'.format(i))
        dim = i + 1
        for j , N in enumerate([10000, 20000, 50000, 80000, 100000]):
            layer_dims = [np.random.randint(60, 80), 100]
            data = gen_data(dim, layer_dims, N).astype('float32')
            eigenvalues = get_eigenvalues(data)
            X[i+j*60,:]=eigenvalues
            y[i+j*60]=dim
    
    np.savez('large_data.npz', X=X, y=y)
