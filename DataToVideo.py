import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

def animate(Num):
    global i
    global SkipFirst
    global count
    global NumFrame
    global im
    global mnist
    global blank_x, blank_y

    if  SkipFirst == True:
        SkipFirst = False
        return [im]
    if Num +1 > NumFrame:
        Num = Num % NumFrame
    if (count > 19) and (Num % NumFrame) == 0:
        i += 9

    print ('Num = %d, Frame = %d' % (Num, i/9))
    a = np.concatenate((mnist[Num][i], blank_x, mnist[Num][i + 1], blank_x, mnist[Num][i + 2]), axis=1)
    b = np.concatenate((mnist[Num][i + 3], blank_x, mnist[Num][i + 4], blank_x, mnist[Num][i + 5]), axis = 1)
    c = np.concatenate((mnist[Num][i + 6], blank_x, mnist[Num][i + 7], blank_x, mnist[Num][i + 8]), axis=1)
    im.set_array( np.concatenate( (a,blank_y,b,blank_y,c),axis = 0) )
    count +=1
    return [im]

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',  comment='Movie support!')
mnist = np.load('mnist_test_seq.npy')
# mnist[20][10000][64][64]
# frame * Numberdata * windowsize_x * windowsize_y
# Assign your data to mnist

ig = plt.figure()
plt.axis("off")
a =(mnist[0][0])
im=plt.imshow(a,interpolation='none',cmap = plt.get_cmap('gray'))

writer = FFMpegWriter(fps=15, metadata=metadata)
NumSample =1
NumFrame = 20

blank_x = np.zeros((64,10))
blank_y = np.zeros((10,212))
SkipFirst = True
i = 0
count = 0
# initialization function: plot the background of each frame
im_ani = manimation.FuncAnimation(ig,animate,NumSample * NumFrame)
im_ani.save('im.mp4', writer=writer)