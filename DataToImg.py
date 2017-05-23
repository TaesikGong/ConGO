import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

def MakeImage(mnist):
	fig = plt.figure()
	plt.axis("off")
	dir_name = "Images"
	if not os.path.exists(dir_name):
		os.makedirs(dir_name) # make directory if not exists
	#im = plt.imshow(dots, interpolation='none', cmap=plt.get_cmap('gray'))
	for i in range(20):
		dots = mnist[i][0]
		plt.imsave('./Images/' + str(i) + '.png',dots,cmap = plt.cm.gray)


def main() :
    return

if __name__ == '__main__' :
    main()