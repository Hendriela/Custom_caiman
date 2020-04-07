"""
Function for preprocessing .tif movies (line shift and correction
as well as draggable colorbar for manual_neuron_selection_gui)
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Calculation of shifts between even and odd lines
# =============================================================================


def shifted_corr(even_lines, odd_lines, lag=0):
    """Calculate correlation between two arrays with a specified lag

    Adrian 2020-03-10
    """

    if lag > 0:
        return np.corrcoef(even_lines[lag:], odd_lines[:-lag])[0, 1]
    elif lag < 0:
        return np.corrcoef(even_lines[:lag], odd_lines[-lag:])[0, 1]
    else:
        return np.corrcoef(even_lines, odd_lines)[0, 1]


def find_shift_image(image, nr_lags=10, debug=False, return_all=False):
    """Get shift between even and odd lines of an image (2d array)

    Adrian 2020-03-10
    """
    lags = np.arange(-nr_lags, nr_lags + 1, 1)
    corr = np.zeros(lags.shape)

    for i, lag in enumerate(lags):
        # pass even and odd lines of the image to the function
        corr[i] = shifted_corr(image[::2, :].flatten(), image[1::2, :].flatten(), lag=lag)

    if debug:
        plt.figure()
        plt.plot(lags, corr)
        plt.title('Maximum at {}'.format(lags[np.argmax(corr)]))

    if not return_all:
        return lags[np.argmax(corr)]  # return only optimal lag
    else:
        # return lags and correlation values
        return lags, corr


def find_shift_stack(stack, nr_lags=10, nr_samples=100, debug=False):
    """Find optimal shift between even and odd lines in stack (nr_frames,x,y)

    Takes nr_samples images from the stack and calculates the lag for values from -nr_lags
    to nr_lags, averages them and gives back the optimal lag between even and odd lines

    Adrian 2020-03-10
    """
    nr_frames = stack.shape[0]

    np.random.seed(123532)
    random_frames = np.random.choice(nr_frames, np.min([nr_samples, nr_frames]), replace=False)
    corrs = list()

    for frame in random_frames:
        lags, corr = find_shift_image(stack[frame, :, :], return_all=True)
        corrs.append(corr)

    avg_corr = np.mean(np.array(corrs), axis=0)  # array from corrs has shape (nr_samples, lags)

    if debug:  # plot avg correlation at various lags with SEM
        plt.figure()
        plt.plot(lags, avg_corr)

        err = np.std(np.array(corrs), axis=0) / np.sqrt(nr_samples)
        m = avg_corr
        plt.fill_between(lags, m - err, m + err, alpha=0.3)
        plt.legend(['Mean', 'SEM'])
        plt.title('Optimal correlation at lag {}'.format(lags[np.argmax(avg_corr)]))

    return lags[np.argmax(avg_corr)]


def apply_shift_to_stack(stack, shift, crop_left=50, crop_right=50):
    """ Shift the corresponding lines (even or odd) to the left to optimal value and crop stack on the left
    Adrian 2020-03-10
    """

    if shift > 0:
        stack[:, ::2, :-shift] = stack[:, ::2, shift:]  # shift all even lines by "shift" to the left
    if shift < 0:
        shift = -shift
        stack[:, 1::2, :-shift] = stack[:, 1::2, shift:]  # shift all odd lines by "shift" to the left

    if crop_left > 0:
        # remove a part on the left side of the image to avoid shifting artifact (if value around 10)
        # or to remove the artifact of late onset of the blanking on Scientifica microscope
        stack = stack[:, :, crop_right:-crop_left]

    return stack


def correct_line_shift_stack(stack, crop_left=5, crop_right=0, nr_samples=100, nr_lags=10):
    """ Correct the shift between even and odd lines in an imaging stack (nr_frames, x, y)

    Adrian 2020-03-10
    """

    line_shift = find_shift_stack(stack, nr_lags=nr_lags, nr_samples=nr_samples)
    print('Correcting a shift of', line_shift, 'pixel.')

    stack = apply_shift_to_stack(stack, line_shift, crop_left=crop_left, crop_right=crop_right)

    return stack

#%% Custom plotting classes for the manual selection GUI


# class from https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib
class Click():
    def __init__(self, axes, func, second_axes = None, button=1):
        self.ax = axes
        self.ax1 = second_axes
        self.func = func
        self.button = button
        self.press = False
        self.move = False
        self.c1 = self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.c2 = self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.c3 = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

    def onclick(self,event):
        # print('Click')
        if event.inaxes == self.ax or event.inaxes == self.ax1 :
            if event.button == self.button:
                # trigger the defined event function
                self.func(event, self.ax, self.ax1)

    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True
    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False


### Class modified from http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
class DraggableColorbar(object):
    def __init__(self, colorbar, axis, image, vmin=0, vmax=1):
        self.cbar = colorbar
        self.mappable = axis
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm,i),'N')])
        self.index = self.cycle.index(self.cbar.get_cmap().name)
        self.img = image
        self.vmin = vmin
        self.vmax = vmax

    def connect(self):
        """connect to all the events we need"""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x,event.y
        #print 'x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx)
        scale = self.vmax - self.vmin
        perc = 0.03
        if event.button==1:
            self.vmin -= (perc*scale)*np.sign(dy)
            self.vmax -= (perc*scale)*np.sign(dy)
        elif event.button==3:
            self.vmin -= (perc*scale)*np.sign(dy)
            self.vmax += (perc*scale)*np.sign(dy)
        self.img.set_clim(self.vmin,self.vmax)
        self.cbar.draw_all()
        # self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()


    def on_release(self, event):
        """on release we reset the press data"""
        self.press = None
        # self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)
