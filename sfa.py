import numpy as np
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 19
import mdp

'''
@param signal_list n x m array with n points in time that are m dimensional
'''
lookback = 6

def train_sfa(signal_list, degree=2, lookback=3):
    #put into format where columns are variables and rows are observations
    ndims = signal_list.shape[1]
    flow = (mdp.nodes.EtaComputerNode() +
        mdp.nodes.TimeFramesNode(lookback) +
        mdp.nodes.PolynomialExpansionNode(degree) +
        mdp.nodes.PCANode(reduce=True)+ 
        mdp.nodes.SFANode(include_last_sample=True, output_dim=6) +
        mdp.nodes.EtaComputerNode() )
    flow.train(signal_list)
    return flow
"""
Just to make sure it's useful, rescale so that the max is 255 and the min is 0
split means split in half
"""
def visualization_matrix(signal, split=False):
    if split:
        return np.hstack([visualization_matrix(signal[:,:3], split=False), visualization_matrix(signal[:,3:],split=False)])
    else:
        return np.interp(signal, (signal.min(), signal.max()), (0, 255))

def make_sfa_node(filename):
    signal = 1.*np.load(filename)
    signal += 0.1*np.random.random(signal.shape)
    print(signal.shape)
    trained_system = train_sfa(signal)
    return trained_system

def plot_im(im):
    im = resize(im,(200,50),order=0,anti_aliasing=True).T
    #im = im.T
    plt.imshow(im, cmap="gray")
    #plt.ylabel("force dimensions")
    plt.xlabel("time")
    #plt.yticks([10,40],["xyz", "rpy"])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    trained_system = make_sfa_node("force_states.npy")
    signal = np.load("force_states.npy")
    im = visualization_matrix(signal,split=True)
    #plot_im(im)
    encoded = trained_system(signal)
    im = visualization_matrix(encoded)
    import ipdb; ipdb.set_trace()
    plot_im(im)
    #Image.fromarray(255*encoded).resize((200,500)).show()







