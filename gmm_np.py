import time
import numpy as np
from matplotlib import pyplot

from scipy import ndimage
from sklearn.cluster import KMeans
from PIL import Image
def initialise_parameters(features, k, d=None, means=None, covariances=None, weights=None):
    """
    Initialises parameters: means, covariances, and mixing probabilities if undefined.

    Arguments:
    features -- input features data set
    k -- number of components in the Gaussian Mixture Model
    """
    if not means or not covariances:
        print("Starting The Initialization")
        val = 250
        n = features.shape[0]
        # Shuffle features set
        indices = np.arange(n)
        np.random.shuffle(indices)
        features_shuffled = features[indices]
        print("Shape of feature_shuffled ", features_shuffled.shape)
        # Estimate means/covariances (or both)
        if not means:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(features.reshape(-1, 1))
            means = kmeans.cluster_centers_
            print("shape of means ", type(means))
            print("Means values ", means)
        if not covariances:
            assert d is not None
            covariances = [val * np.identity(d) for i in range(k)]
            
    if not weights:
        weights = np.ones(k) / k

    return (np.asarray([[100], [70]]), covariances, weights)

# gaussian function
def gau(mean, var, varInv, feature, d):
    var_det = np.linalg.det(var)
    a = np.sqrt(((2 * (np.pi))**d) * var_det)
    b = np.exp(-0.5 * np.dot((feature - mean), np.dot(varInv, (feature - mean).T)))
    return b / a

def smsi_likeli(mean, var, varInv, s_value, feature, d):
    temp = []
    for x in range(k):
        temp.append(s_value * gau(mean[x], var[x], varInv[x], feature, d))
    return np.array(temp)

def saliency(img):
    c = np.fft.fft2(img)
    d = c
    print("shape of c :", c.shape)
    print(c)
    phase_spectrum = np.angle(c)
    mag = np.abs(c)
    spectralResidual = np.exp(np.log(mag) - ndimage.uniform_filter(np.log(mag), size=5))
    
    c[:, :] = c[:, :] * spectralResidual / mag
    c = np.fft.ifft2(c).real
    mag = np.abs(c)
    mag = ndimage.gaussian_filter(mag, sigma=3)
    mag = (mag - mag.min()) / (mag.max() - mag.min())
    pyplot.subplot(2, 2, 2)
    pyplot.imshow( 10000 * np.exp(mag), cmap='gray')
    pyplot.show()
    print("shape of mag ", mag.shape)
    return mag

def neighbor_prob(img, fun):
    mask = np.ones((3, 3))
    result = ndimage.generic_filter(img, function=fun, footprint=mask, mode='constant', cval=np.NaN)
    return result

def sliding_window(im):
    rows, cols = im.shape
    final = np.zeros((rows, cols, 3, 3))
    for x in (0, 1, 2):
        for y in (0, 1, 2):
            im1 = np.vstack((im[x:], im[:x]))
            im1 = np.column_stack((im1[:, y:], im1[:, :y]))
            final[x::3, y::3] = np.swapaxes(im1.reshape(int(rows / 3), 3, int(cols / 3), -1), 1, 2)
    return final

input_path = "STARfish.jpg"

# Read the image using Pillow
img = Image.open(input_path).convert("L")  # Convert to grayscale
img = np.array(img)  # Convert to numpy array
print("img shape ", img.shape)

pyplot.subplot(2, 2, 1)
pyplot.imshow(img, cmap='gray')# no of clusters
pyplot.show()
print("Image can be now seen")
print("Image successfully taken as input")

k = 2
o_shape = img.shape

# Gray scale
d = 1

# Array of pixels
feat = img.reshape(-1)
print("Shape of feat ", feat.shape)

# Total no of pixels
N = len(feat)

# s = saliency(input_path)
s = saliency(img)

means, covariances, weights = initialise_parameters(features=feat, k=k, d=d)
covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
meanPrev = [np.array([0]) for i in range(k)] #what is this mean previous doing
iteration = []
logLikelihoods = []
counterr = 0
smsi_init_weights = np.ones((o_shape[0] * o_shape[1], k))

#why initial weights are like this?

# Ri value
R_value = neighbor_prob(s, np.nansum)
R_value = R_value.reshape(-1)
s_value = s.reshape(-1)

while abs((np.asarray(meanPrev) - np.asarray(means)).sum()) > 0.001:
    start = time.time()
    smsi_resp = []
    for i, feature in enumerate(feat):
        smsi_classLikelihoods = smsi_likeli(means, covariances, covariances_Inv, s_value[i], feature, d)
        smsi_resp.append(smsi_classLikelihoods)

    smsi_resp = np.asarray(smsi_resp)

    final_resp = []
    for cluster in range(k):
        test = smsi_resp[:, cluster]
        result = ndimage.generic_filter(test.reshape(o_shape[0], o_shape[1]), np.nansum, footprint=np.ones((3, 3)),
                                        mode='constant', cval=np.NaN).reshape(-1)
        final_resp.append(result * smsi_init_weights[:, cluster] / R_value)

    # numerator of gama
    smsi_resp_num = np.asarray(final_resp).T

    # denominator of gama
    final_resp_den = smsi_resp_num.sum(axis=1)

    # gama value of the smsi
    final_smsi_resp = []
    for cluster in range(k):
        final_smsi_resp.append(smsi_resp_num[:, cluster] / final_resp_den)

    final_smsi_resp = np.asarray(final_smsi_resp).T

    # SMSI MEAN
    smsi_mean_den = final_smsi_resp.sum(axis=0)
    smsi_mean_num_s_x = s_value * feat
    smsi_mean_num = []
    smsi_mean_num_s_x = ndimage.generic_filter(smsi_mean_num_s_x.reshape(o_shape[0], o_shape[1]), np.nansum,
                                               footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
    for i in range(k):
        smsi_mean_num.append(final_smsi_resp[:, i] * (smsi_mean_num_s_x / R_value))
    smsi_mean_num = np.asarray(smsi_mean_num).T
    smsi_mean = (smsi_mean_num.sum(axis=0) / smsi_mean_den).T
    meanPrev = means
    means = smsi_mean

    # CO VAR
    f_u = np.asarray([feat - means[i] for i in range(k)]).T

    segmentedImage = np.zeros((N), np.uint8)

for i, resp in enumerate(final_smsi_resp):
    max = resp.argmax()
    print(means)
    segmentedImage[i] = 255 - 255 * means[max]
    
segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])
pyplot.subplot(2, 2, 3)
pyplot.imshow(segmentedImage, cmap='gray')

# Show the plot
pyplot.show()