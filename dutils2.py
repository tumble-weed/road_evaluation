import model.utils 
ROOT_DIR = 'debug-results'
import os
img_save = lambda im,filename,ROOT_DIR=ROOT_DIR:model.utils.img_save(im,os.path.join(ROOT_DIR,filename))

save_plot = lambda y,title,filename,ROOT_DIR=ROOT_DIR:model.utils.save_plot(y,title,os.path.join(ROOT_DIR,filename))
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def array_info(*args):
    for ar in args:
        try:
            print(ar.__class__)
            print(ar.shape)
            print(ar.min())
            print(ar.max())
            print('*'*10)
        except Exception as e:
            print('exception')
    pass

def show_pyramids(gpnn_inpainting,pi):
    import inspect
    # v =  inspect.currentframe().f_back.f_locals
    # gpnn_inpainting = v['gpnn_inpainting']
    try:
        img_save(tensor_to_numpy(gpnn_inpainting.mask_pyramid[pi][0,:,:,0]),'mask_pyramid.png')
        img_save(tensor_to_numpy(gpnn_inpainting.x_pyramid[pi][0,:,:,0]),'x_pyramid.png')
        img_save(tensor_to_numpy(gpnn_inpainting.y_pyramid[pi][...,:3][0]),'y_pyramid.png')
    except Exception as e:
        print(e)

def UPDATE_PAPER(d):
    pass
#========================================
import IPython.core.ultratb
import sys
def e():
    tb = IPython.core.ultratb.VerboseTB()
    print(tb.text(*sys.exc_info()))
#========================================
def list_info(*args):
    for ar in args:
        try:
            print(ar.__len__())
            if len(ar):
                print(ar[0].__class__)
        except Exception as e:
            print('exception')
    pass