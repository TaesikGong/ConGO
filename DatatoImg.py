import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

def dataToImage():
    fig = plt.figure()
    mnist = np.load('mnist_test_seq.npy') # (20, 10000, 64, 64)

    # mnist is the name of the image
    # Assign your data to mnist

    plt.axis("off")

    #im = plt.imshow(dots, interpolation='none', cmap=plt.get_cmap('gray'))
    for i in range(20):
        dots = mnist[i][0]
        plt.imsave('./Images/' + str(i) + '.png',dots,cmap = plt.cm.gray)


def main() :
    dataToImage()

if __name__ == '__main__' :
    main()