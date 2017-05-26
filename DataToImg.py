import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

def MakeImage(mnist,name):
	fig = plt.figure()
	plt.axis("off")
	dir_name = "Images"
	if not os.path.exists(dir_name):
		os.makedirs(dir_name) # make directory if not exists
	#im = plt.imshow(dots, interpolation='none', cmap=plt.get_cmap('gray'))
	dots = mnist[0]
	for i in range(1,20):
		dots = np.concatenate((dots,mnist[i]),axis=1)
 	plt.imsave('./Images/' +i+'_'+ name +'.png',dots,cmap = plt.cm.gray)


def main() :
    return

if __name__ == '__main__' :
    main()
