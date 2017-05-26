import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import sys
import os
NumSample = 1
NumFrame = 20
blank_x = np.ones((64, 10)) * 255
blank_y = np.ones((10, 212)) * 255
SkipFirst = True
i = 0
count = 0

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')


def MakeVideo(mnist,index,dir_name):    
    # mnist[20][10000][64][64]
    # frame * Numberdata * windowsize_x * windowsize_y
    # initialization function: plot the background of each frame
#    mnist = np.load('mnist_test_seq.npy')	
	if not os.path.exists(dir_name):
		os.makedirs(dir_name) # make directory if not exists
	ig = plt.figure()
	plt.axis("off")        
	writer = FFMpegWriter(fps=5, metadata=metadata)
    #initarray = mnist[0]
	loss = str(mnist[2])
	initarray = np.concatenate( (mnist[0][0],mnist[1][0]),axis= 1)
	im = plt.imshow(initarray, interpolation='none', cmap=plt.get_cmap('gray'))    
	im_ani = manimation.FuncAnimation(ig,animate,NumSample * NumFrame,fargs = (im,mnist))
	im_ani.save(dir_name+str(index)+'_output_loss_' + loss + '_.mp4', writer=writer)		

def animate(Num,im,mnist):
    global i
    global SkipFirst
    global count
    global NumFrame
    #global im
    global blank_x, blank_y
    if  (SkipFirst == True):
        SkipFirst = False
        return [im]
    sumarray = np.concatenate( (mnist[0][Num], blank_x, mnist[1][Num]), axis = 1 )
    #sumarray = mnist[Num]
    im.set_array(sumarray)
    return [im]
'''
    if  SkipFirst == True:
        SkipFirst = False
        return [im]
    if Num +1 > NumFrame:
        Num = Num % NumFrame
    if (count > 19) and (Num % NumFrame) == 0:
        i += 9
    
'''
'''
    print ('Num = %d, Frame = %d' % (Num, i/9))
    a = np.concatenate((mnist[Num][i], blank_x, mnist[Num][i + 1], blank_x, mnist[Num][i + 2]), axis=1)
    b = np.concatenate((mnist[Num][i + 3], blank_x, mnist[Num][i + 4], blank_x, mnist[Num][i + 5]), axis = 1)
    c = np.concatenate((mnist[Num][i + 6], blank_x, mnist[Num][i + 7], blank_x, mnist[Num][i + 8]), axis=1)
    im.set_array( np.concatenate( (a,blank_y,b,blank_y,c),axis = 0) )
'''


def main() :
    return

if __name__ == '__main__' :
    main()