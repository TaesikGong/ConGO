import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

def MakeImage(mnist):
    fig = plt.figure()
    plt.axis("off")

    #im = plt.imshow(dots, interpolation='none', cmap=plt.get_cmap('gray'))
    for i in range(20):
        dots = mnist[i][0]
        plt.imsave('./Images/' + str(i) + '.png',dots,cmap = plt.cm.gray)


def main() :
    return

if __name__ == '__main__' :
    main()